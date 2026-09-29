# -*- coding: utf-8 -*-
"""
KVASIR ZERO-DICE DIAGNOSTIC
===========================
Goal: explain the ~350 images with Dice == 0. This reuses your EXACT pipeline
(imported from LENAS_Kvasir_revised.py) and, for each image, records where the
mask is and where it dies, then categorizes every failure.

It answers, with numbers:
  * Are some GROUND-TRUTH masks empty / tiny / mis-loaded?   (empty_gt)
  * Does SAM return an empty mask?                            (sam_empty)
  * Does the SNAKE step erase the mask?                       (snake_empty)
  * Does the ITERATIVE loop erase / move the mask?            (iter_empty / no_overlap)
  * Does CRF erase it?                                        (crf_empty)
  * Do the POSITIVE prompts even land inside the polyp GT?    (prompt_in_gt rate)

"""
import os, sys, csv
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------------------- CONFIG
CONFIG = {
    # copy these four from your LENAS_Kvasir_revised.py main():
    "KVASIR_IMAGES": "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/images",
    "KVASIR_MASKS":  "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/masks",
    "SAM_CKPT":      "/media/data/DARE/sam_vit_b_01ec64.pth",
    "CAPSULE_CKPT":  "best_kvasir_domain_classifier.pth",  # the in-domain classifier you just trained

    "MODEL_NAME": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "MODULE_NAME": "LENAS_Kvasir_revised",   # the file to import (no .py)
    "MAX_SAMPLES": 300,        # set None for all 1000 (slower). 300 is enough to categorize.
    "USE_ITERATIVE": True,     # match your run
    "FUSION_STRATEGY": "weighted_average",
    # Set to a list to compare fusion methods (reviewer ask). None = single run.
    # Valid: weighted_average, multiplicative_consensus, weighted_geometric_mean, rank_aggregation
    "SWEEP_FUSION": None,   # set to a list to compare fusion methods; None = single fast run
    "XAI_WEIGHTS": {"Saliency": 0.3983, "IntegratedGradients": 0.2987, "GradientShap": 0.3030},
    "OUT_CSV": "./LENAS_outputs_v2/results/kvasir_zero_dice_diagnostic.csv",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Fallback labels only; build_model() prefers the class list stored INSIDE the checkpoint.
CAPSULE_CLASS_LABELS = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
                        'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps',
                        'ulcerative-colitis']

# ----------------------------------------------------------------------------- import pipeline
import importlib, inspect
L = importlib.import_module(CONFIG["MODULE_NAME"])
try:
    L = importlib.reload(L)   # always pick up the latest on-disk version
except Exception as _e:
    print(f"[reload warning] {_e}")
print(f"Imported pipeline module from: {getattr(L, '__file__', '?')}", flush=True)
_has_initmask = "init_mask" in inspect.signature(L.iterative_self_correction_improved).parameters
_has_guard = "KVASIR_SNAKE_NONEROSION" in getattr(L, "REVISION_CONFIG", {})
print(f"[version check] iterative.init_mask={_has_initmask} | snake_guard_cfg={_has_guard} | "
      f"candidate_sel_fn={hasattr(L,'select_best_sam_candidate')}", flush=True)
if not _has_initmask:
    print("  *** STALE MODULE: on-disk LENAS_Kvasir_revised.py is OLD (no init_mask). "
          "Save the NEW file to disk and RESTART the kernel. ***", flush=True)

def area(m):
    return int((np.asarray(m) > 0).sum())

def build_model():
    ckpt = torch.load(CONFIG["CAPSULE_CKPT"], map_location=DEVICE, weights_only=False)
    # Prefer the class list/count stored in the checkpoint so the head always matches.
    if isinstance(ckpt, dict) and "classes" in ckpt:
        classes = list(ckpt["classes"])
    else:
        classes = CAPSULE_CLASS_LABELS
    num_classes = int(ckpt.get("num_classes", len(classes))) if isinstance(ckpt, dict) else len(classes)
    print(f"Building model with {num_classes} classes: {classes}")
    m = L.DifferentialBiomedCLIP(CONFIG["MODEL_NAME"], num_classes, DEVICE,
                                 class_names=classes, use_contrastive=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = m.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint OK (val_acc={ckpt.get('val_acc') if isinstance(ckpt, dict) else 'n/a'}, "
          f"polyp_idx={ckpt.get('polyp_idx') if isinstance(ckpt, dict) else 'n/a'})")
    # Force the XAI target class to the polyp index stored in the checkpoint, so a stale
    # module config cannot point the saliency at the wrong class.
    if isinstance(ckpt, dict) and ckpt.get("polyp_idx") is not None:
        L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = True
        L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(ckpt["polyp_idx"])
        print(f"[target] forced KVASIR_TARGET_CLASS = {ckpt['polyp_idx']} from checkpoint")
    m.to(DEVICE).eval()
    return m

def build_sam():
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=CONFIG["SAM_CKPT"]).to(DEVICE)
        print("SAM loaded.")
        return SamPredictor(sam)
    except Exception as e:
        print(f"SAM load failed: {e}")
        return None

def diagnose_one(sample, model, sam_predictor):
    """Mirror of _kvasir_segment_metrics_only but records every intermediate area."""
    image_tensor, gt_mask_tensor, img_path = sample
    original_np = np.array(Image.open(img_path).convert("RGB"))
    gt = (gt_mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    rec = {"img": os.path.basename(img_path),
           "gt_h": gt.shape[0], "gt_w": gt.shape[1],
           "img_h": original_np.shape[0], "img_w": original_np.shape[1],
           "gt_area": area(gt), "gt_frac": round(area(gt) / gt.size, 5)}

    with torch.no_grad():
        _, entropy, probs = model.predict_with_uncertainty(image_tensor.unsqueeze(0).to(DEVICE))
        predicted_class = int(torch.argmax(probs, dim=1).item())
        entropy_val = float(entropy[0].item())
    target_class = L.REVISION_CONFIG.get("KVASIR_TARGET_CLASS", 7) \
        if L.REVISION_CONFIG.get("KVASIR_FORCE_POLYP_CLASS", True) else predicted_class
    rec["pred_class"] = predicted_class
    rec["target_class"] = target_class

    expl = L.generate_explanations_focused(model, image_tensor.unsqueeze(0), target_class, DEVICE, use_cache=False)
    fused = L.advanced_xai_fusion(expl, CONFIG["XAI_WEIGHTS"], CONFIG["FUSION_STRATEGY"])
    if L.REVISION_CONFIG.get("SUPPRESS_BORDER_SPECULAR", True) and hasattr(L, "saliency_validity_mask"):
        v = L.saliency_validity_mask(original_np)
        vx = cv2.resize(v, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_NEAREST)
        fm = fused * (vx > 0)
        if fm.max() > 0:
            fused = fm
    if L.REVISION_CONFIG.get("XAI_SHARPEN", True) and hasattr(L, "sharpen_saliency_map"):
        fused = L.sharpen_saliency_map(fused)
    rec["fused_max"] = round(float(fused.max()), 4)
    rec["fused_mean"] = round(float(fused.mean()), 5)

    num_prompts, _ = L.uncertainty_guided_prompts(fused, entropy_val, probs[0].cpu().numpy())
    bbox = L.extract_focused_bbox_from_saliency(fused, top_k_percent=0.05)
    pos = L.extract_focused_positive_prompts(fused, bbox, num_prompts=num_prompts)
    circ = L.detect_circular_mask(original_np.shape)
    circ_x = cv2.resize(circ, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_NEAREST)
    neg = L.extract_smart_negative_prompts(fused, bbox, circ_x, num_negatives=3)

    xs, os_ = fused.shape, original_np.shape[:2]
    bbox_o = L.transform_bbox_to_original(bbox, xs, os_)
    pos_o = L.transform_coordinates_to_original(pos, xs, os_)
    neg_o = L.transform_coordinates_to_original(neg, xs, os_) if len(neg) > 0 else np.array([])

    # do the prompts land inside GT?
    def in_gt(pts):
        c = 0
        for p in np.atleast_2d(pts):
            if p.size < 2:
                continue
            x, y = int(p[0]), int(p[1])
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1] and gt[y, x] > 0:
                c += 1
        return c
    rec["n_pos"] = int(len(np.atleast_2d(pos_o))) if pos_o.size else 0
    rec["n_pos_in_gt"] = in_gt(pos_o) if pos_o.size else 0
    rec["n_neg"] = int(len(np.atleast_2d(neg_o))) if neg_o.size else 0
    rec["n_neg_in_gt"] = in_gt(neg_o) if neg_o.size else 0
    # bbox overlap with GT (IoU of boxes is hard; use: does GT centroid fall in bbox?)
    if rec["gt_area"] > 0:
        ys, xsd = np.where(gt > 0)
        cy, cx = ys.mean(), xsd.mean()
        x0, y0, x1, y1 = bbox_o
        rec["gt_center_in_bbox"] = int(x0 <= cx <= x1 and y0 <= cy <= y1)
    else:
        rec["gt_center_in_bbox"] = -1

    sam_raw = snake = iter_final = final = np.zeros(gt.shape, np.uint8)
    if sam_predictor is not None:
        if L.REVISION_CONFIG.get("KVASIR_CANDIDATE_SELECTION", True) and hasattr(L, "select_best_sam_candidate"):
            sam_raw, _ = L.select_best_sam_candidate(model, sam_predictor, original_np, image_tensor,
                                                     fused, target_class, DEVICE, bbox_o, pos_o, neg_o)
        else:
            sam_raw, _ = L.segment_with_sam_enhanced(sam_predictor, original_np, pos_o, neg_o, bbox_o, post_process=True)
        if L.REVISION_CONFIG.get("KVASIR_RECALL_GROWTH", True) and hasattr(L, "grow_mask_to_saliency"):
            sam_raw = L.grow_mask_to_saliency(sam_predictor, sam_raw, fused, original_np,
                                              model, image_tensor, target_class, DEVICE,
                                              explanations=expl)
        snake = L.refine_mask_with_snake_improved(sam_raw, original_np, fused, iterations=100)
        if CONFIG["USE_ITERATIVE"]:
            # Pass the pipeline's snake mask so iterative REFINES it (progressive) instead
            # of recomputing from scratch. Falls back gracefully on older module versions.
            try:
                iter_final, _ = L.iterative_self_correction_improved(
                    model, sam_predictor, original_np, image_tensor, target_class,
                    DEVICE, CONFIG["XAI_WEIGHTS"],
                    max_iterations=L.REVISION_CONFIG.get("TMAX", 5), init_mask=snake)
            except TypeError:
                print("  [warn] module has no init_mask param -> using OLD (restart) iterative. "
                      "Update LENAS_Kvasir_revised.py to test the fix.")
                iter_final, _ = L.iterative_self_correction_improved(
                    model, sam_predictor, original_np, image_tensor, target_class,
                    DEVICE, CONFIG["XAI_WEIGHTS"], max_iterations=L.REVISION_CONFIG.get("TMAX", 5))
            final = iter_final
        else:
            final = snake
    crf = L.maybe_crf(original_np, final)
    # Mirror the pipeline's stage-selection so 'final' reflects the real output.
    if sam_predictor is not None and L.REVISION_CONFIG.get("KVASIR_STAGE_SELECT", True) and hasattr(L, "select_best_stage_mask"):
        final = L.select_best_stage_mask(model, image_tensor, target_class, DEVICE,
                                         [sam_raw, snake, iter_final, crf],
                                         (original_np.shape[0], original_np.shape[1]))
    else:
        final = crf

    rec["sam_raw_area"] = area(sam_raw)
    rec["snake_area"] = area(snake)
    rec["iter_area"] = area(iter_final)
    rec["final_area"] = area(final)
    fb = (np.asarray(final) > 0).astype(np.uint8)
    rec["overlap"] = int((fb & gt).sum()) if gt.shape == fb.shape else -1
    rec["dice"] = round(float(L.calculate_dice_score(fb, gt)), 4)
    rec["iou"] = round(float(L.calculate_iou_score(fb, gt)), 4)
    # Per-stage Dice vs GT -- shows whether the cap is the bbox (low SAM_raw) or
    # downstream erosion (SAM_raw high but snake/iter/CRF drop it).
    def _dvg(_m):
        _mb = (np.asarray(_m) > 0).astype(np.uint8)
        return round(float(L.calculate_dice_score(_mb, gt)), 4) if (gt.shape == _mb.shape) else -1.0
    rec["dice_sam"] = _dvg(sam_raw)
    rec["dice_snake"] = _dvg(snake)
    rec["dice_iter"] = _dvg(iter_final)
    rec["dice_final"] = rec["dice"]
    # Per-stage precision / recall vs GT (recall drop => erosion; precision drop => added background)
    def _pr(_m):
        _mb = (np.asarray(_m) > 0).astype(np.uint8)
        if _mb.shape != gt.shape:
            return (-1.0, -1.0)
        tp = int(np.logical_and(_mb, gt).sum())
        fp = int(np.logical_and(_mb, 1 - gt).sum())
        fn = int(np.logical_and(1 - _mb, gt).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (round(prec, 4), round(rec, 4))
    for _nm, _mk in [("sam", sam_raw), ("snake", snake), ("iter", iter_final), ("final", final)]:
        _p, _r = _pr(_mk)
        rec[f"prec_{_nm}"] = _p
        rec[f"rec_{_nm}"] = _r

    # category
    if rec["gt_area"] == 0:
        cat = "empty_gt"
    elif rec["sam_raw_area"] == 0:
        cat = "sam_empty"
    elif rec["final_area"] == 0:
        cat = "refine_erased"          # SAM had a mask but snake/iter/crf erased it
    elif rec["overlap"] == 0:
        cat = "no_overlap"             # mask exists but in the wrong place
    elif rec["dice"] < 0.1:
        cat = "tiny_overlap"
    else:
        cat = "ok"
    rec["category"] = cat
    return rec

def main():
    print("Device:", DEVICE)
    # ---- (0) GT-ONLY SANITY (fast, no model): are some masks empty/tiny? ----
    imgs = sorted([f for f in os.listdir(CONFIG["KVASIR_IMAGES"]) if f.endswith(".jpg")])
    _od = getattr(L, "REVISION_CONFIG", {}).get("OUTPUT_DIR")
    if _od:
        CONFIG["OUT_CSV"] = os.path.join(_od, "results", "kvasir_zero_dice_diagnostic.csv")
    print(f"[output] CSV target: {CONFIG['OUT_CSV']}", flush=True)
    print(f"\n[GT sanity] {len(imgs)} images found.")
    empty_gt = tiny_gt = 0
    fracs = []
    for f in imgs:
        mp = os.path.join(CONFIG["KVASIR_MASKS"], f)
        if not os.path.exists(mp):
            empty_gt += 1; continue
        m = np.array(Image.open(mp).convert("L"))
        fg = (m > 127).mean()
        fracs.append(fg)
        if fg == 0:
            empty_gt += 1
        elif fg < 0.005:
            tiny_gt += 1
    fracs = np.array(fracs)
    print(f"[GT sanity] empty GT masks: {empty_gt} | tiny (<0.5% fg): {tiny_gt}")
    if len(fracs):
        print(f"[GT sanity] GT foreground fraction  min={fracs.min():.4f} "
              f"med={np.median(fracs):.4f} max={fracs.max():.4f}")

    # ---- (1) full pipeline categorization ----
    model = build_model()
    sam = build_sam()
    xform = model.preprocess
    mask_xform = transforms.Compose([transforms.ToTensor()])
    ds = L.KvasirSEGDataset(CONFIG["KVASIR_IMAGES"], CONFIG["KVASIR_MASKS"],
                            transform=xform, mask_transform=mask_xform)
    n = len(ds) if CONFIG["MAX_SAMPLES"] is None else min(CONFIG["MAX_SAMPLES"], len(ds))
    _tc = L.REVISION_CONFIG.get("KVASIR_TARGET_CLASS", 7)
    print(f"\n[pipeline] KVASIR_TARGET_CLASS = {_tc}  (must be the polyps index, e.g. 6 for the 8-class model)")
    print(f"[pipeline] analyzing {n} samples...")
    def analyze(tag=""):
        recs = []
        for i in range(n):
            item = ds[i]
            if item is None or item[0] is None:
                continue
            try:
                recs.append(diagnose_one(item, model, sam))
            except Exception as e:
                print(f"  sample {i} failed: {e}")
            if (i + 1) % 25 == 0:
                print(f"   [{tag}] {i+1}/{n} done")
                torch.cuda.is_available() and torch.cuda.empty_cache()
        return recs

    # ---- optional fusion-strategy sweep (reviewer: compare fusion methods) ----
    sweep = CONFIG.get("SWEEP_FUSION")
    if sweep:
        print(f"\n[fusion sweep] comparing {len(sweep)} strategies on {n} samples...")
        sweep_res = {}
        for strat in sweep:
            CONFIG["FUSION_STRATEGY"] = strat
            r = analyze(tag=strat)
            d = np.array([x["dice"] for x in r]) if r else np.array([0.0])
            io = np.array([x["iou"] for x in r]) if r else np.array([0.0])
            sweep_res[strat] = (float(d.mean()), float(io.mean()), int((d < 1e-6).sum()), len(r))
        print("\n" + "=" * 64)
        print("FUSION STRATEGY COMPARISON")
        print("=" * 64)
        print(f"{'strategy':>24} | {'meanDice':>9} | {'meanIoU':>8} | {'zeros':>9}")
        for s, (md, mi, z, ns) in sorted(sweep_res.items(), key=lambda kv: -kv[1][0]):
            print(f"{s:>24} | {md:9.4f} | {mi:8.4f} | {z:>4}/{ns}")
        best = max(sweep_res, key=lambda s: sweep_res[s][0])
        print(f"\nBest fusion strategy: {best}  (Dice {sweep_res[best][0]:.4f})")
        print("=" * 64)
        CONFIG["FUSION_STRATEGY"] = best   # detailed run + CSV use the best strategy

    print(f"\n[detailed run] fusion = {CONFIG['FUSION_STRATEGY']}")
    records = analyze(tag=CONFIG["FUSION_STRATEGY"])
    if not records:
        print("No records produced."); return

    os.makedirs(os.path.dirname(CONFIG["OUT_CSV"]) or ".", exist_ok=True)
    with open(CONFIG["OUT_CSV"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    dices = np.array([r["dice"] for r in records])
    cats = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    zeros = [r for r in records if r["dice"] < 1e-6]
    zero_cats = {}
    for r in zeros:
        zero_cats[r["category"]] = zero_cats.get(r["category"], 0) + 1
    pos_in_gt_rate = np.mean([r["n_pos_in_gt"] > 0 for r in records])
    neg_in_gt_rate = np.mean([r["n_neg_in_gt"] > 0 for r in records])
    bbox_hit = np.mean([r["gt_center_in_bbox"] == 1 for r in records if r["gt_center_in_bbox"] >= 0])

    print("\n" + "=" * 64)
    print("DIAGNOSTIC SUMMARY  (paste this whole block back)")
    print("=" * 64)
    print(f"samples analyzed        : {len(records)}")
    print(f"mean Dice               : {dices.mean():.4f}")
    print(f"Dice == 0 count         : {len(zeros)}  ({100*len(zeros)/len(records):.1f}%)")
    print(f"category counts (all)   : {cats}")
    print(f"category counts (zeros) : {zero_cats}")
    print(f">=1 positive prompt in GT: {100*pos_in_gt_rate:.1f}% of images")
    print(f">=1 negative prompt in GT: {100*neg_in_gt_rate:.1f}% of images  (high = SAM sabotaged)")
    print(f"GT center inside bbox    : {100*bbox_hit:.1f}% of images")
    print("--- stage where mask dies (mean area, px) ---")
    for k in ["gt_area", "sam_raw_area", "snake_area", "iter_area", "final_area"]:
        print(f"   {k:>14}: {np.mean([r[k] for r in records]):10.1f}")
    print("--- mean Dice at each stage vs GT (where the score is lost) ---")
    for k, lab in [("dice_sam", "SAM_raw"), ("dice_snake", "snake"),
                   ("dice_iter", "iterative"), ("dice", "final_CRF")]:
        vals = [r[k] for r in records if r.get(k, -1) is not None and r.get(k, -1) >= 0]
        if vals:
            print(f"   {lab:>10}: {np.mean(vals):.4f}")
    print("--- per-stage precision / recall vs GT ---")
    for nm, lab in [("sam", "SAM_raw"), ("snake", "snake"), ("iter", "iterative"), ("final", "final_CRF")]:
        ps = [r[f"prec_{nm}"] for r in records if r.get(f"prec_{nm}", -1) >= 0]
        rs = [r[f"rec_{nm}"] for r in records if r.get(f"rec_{nm}", -1) >= 0]
        if ps:
            print(f"   {lab:>10}: precision {np.mean(ps):.3f} | recall {np.mean(rs):.3f}")
    print("--- stage transitions (mean dDice; cause of drops) ---")
    def _transition(a, b, la, lb):
        dd, rec_driven, prec_driven, worse, better = [], 0, 0, 0, 0
        for r in records:
            da, db = r.get(f"dice_{a}", -1), r.get(f"dice_{b}", -1)
            if da < 0 or db < 0:
                continue
            delta = db - da
            dd.append(delta)
            if delta < -0.02:
                worse += 1
                dr = r.get(f"rec_{a}", 0) - r.get(f"rec_{b}", 0)     # recall lost
                dp = r.get(f"prec_{a}", 0) - r.get(f"prec_{b}", 0)   # precision lost
                if dr >= dp:
                    rec_driven += 1
                else:
                    prec_driven += 1
            elif delta > 0.02:
                better += 1
        if dd:
            cause = "erosion/recall-loss" if rec_driven >= prec_driven else "added-background/precision-loss"
            print(f"   {la:>8} -> {lb:<9}: mean dDice {np.mean(dd):+.4f} | worse {worse} (mostly {cause}: "
                  f"{rec_driven} recall vs {prec_driven} precision) | better {better}")
    _transition("sam", "snake", "SAM", "snake")
    _transition("snake", "iter", "snake", "iter")
    _transition("iter", "final", "iter", "final")
    print(f"CSV written -> {CONFIG['OUT_CSV']}")
    print("=" * 64)

if __name__ == "__main__":
    main()
