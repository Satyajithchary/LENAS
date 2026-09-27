# -*- coding: utf-8 -*-
"""
LENAS BUSI RUNNER
=================
Runs the FULL, already-debugged LENAS pipeline (from LENAS_Kvasir_revised.py) on the BUSI
breast-ultrasound dataset. Every Kvasir improvement -- SmoothGrad + sharpening, reward-based
candidate selection, blob coverage, annotation-free recall growth with per-XAI selection, the
snake non-erosion guard, and the iterative init_mask fix -- is reused unchanged (they are
dataset-agnostic module functions gated by REVISION_CONFIG, defaults on).

BUSI difference from Kvasir: there is no single target class. Each image's XAI target is its
OWN lesion class (benign / malignant); 'normal' images have no lesion/mask and are skipped.
This stays annotation-free for segmentation (image-level class label only, no pixel labels).

Prereqs
-------
1. Train a BUSI in-domain classifier first (image-level labels only):
     edit train_lenas_classifier.py -> MODE="imagefolder", DATA_DIR=<BUSI root>,
     SAVE_PATH="best_busi_domain_classifier.pth"; run it. Note the printed class list.
2. Keep the NEW LENAS_Kvasir_revised.py and the SAM checkpoint on disk.

Usage
-----
  python lenas_busi_run.py --smoke        # quick test on a few images
  python -u lenas_busi_run.py 2>&1 | tee busi_run.log     # full run
"""
import os, sys, argparse, importlib, inspect
import numpy as np

# ----------------------------------------------------------------- CONFIG
CONFIG = {
    "MODULE_NAME": "LENAS_Kvasir_revised",   # reuse the debugged pipeline
    "MODEL_NAME": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "BUSI_DATA_DIR": "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",  # benign/ malignant/ normal/
    "CLASSIFIER_CKPT": "best_busi_domain_classifier.pth",
    "SAM_CKPT": "/media/data/DARE/sam_vit_b_01ec64.pth",
    "OUT_CSV": "./LENAS_BUSI_outputs/results/busi_segmentation.csv",
    "XAI_WEIGHTS": {"Saliency": 0.279, "IntegratedGradients": 0.352, "GradientShap": 0.369},
    "FUSION_STRATEGY": "weighted_average",
    "USE_ITERATIVE": True,
    "MAX_SAMPLES": None,      # None = all benign+malignant; int caps it
    "SKIP_NORMAL": True,      # 'normal' has no lesion mask -> exclude from segmentation eval
}

# BUSI over-segments (unlike Kvasir); turn off the growth/blob/CRF steps that expand masks.
BUSI_OVERRIDES = {
    # Best config from the BUSI diagnostic sweep. BUSI ultrasound OVER-segments (opposite of
    # Kvasir), so the Kvasir reward candidate-selection / growth / blob steps HURT it (benign
    # precision 0.18 -> 0.33 when disabled). CRF also hurts. Use the plain pipeline + GT label.
    "KVASIR_CANDIDATE_SELECTION": False,
    "KVASIR_RECALL_GROWTH": False,
    "KVASIR_BLOB_COVERAGE": False,
    "USE_CRF": False,
    "KVASIR_STAGE_SELECT": False,
}


def load_pipeline():
    L = importlib.reload(importlib.import_module(CONFIG["MODULE_NAME"]))
    print(f"[pipeline] module: {getattr(L,'__file__','?')}", flush=True)
    ok = "init_mask" in inspect.signature(L.iterative_self_correction_improved).parameters
    print(f"[version check] iterative.init_mask={ok} | candidate_sel={hasattr(L,'select_best_sam_candidate')} | "
          f"growth={hasattr(L,'grow_mask_to_saliency')}", flush=True)
    if not ok:
        print("  *** STALE LENAS_Kvasir_revised.py (no init_mask). Save the new file & restart. ***", flush=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(L.resolve_ckpt(CONFIG["CLASSIFIER_CKPT"]) if hasattr(L, "resolve_ckpt")
                      else CONFIG["CLASSIFIER_CKPT"], map_location=device, weights_only=False)
    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    ncls = int(ckpt.get("num_classes", len(classes) if classes else 3)) if isinstance(ckpt, dict) else 3
    model = L.DifferentialBiomedCLIP(CONFIG["MODEL_NAME"], ncls, device,
                                     class_names=classes, use_contrastive=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt,
                          strict=True)
    model.to(device).eval()
    print(f"[classifier] {ncls} classes: {classes}", flush=True)

    sam = None
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=CONFIG["SAM_CKPT"]).to(device))
        print("[pipeline] SAM loaded.", flush=True)
    except Exception as e:
        print(f"[pipeline] SAM unavailable: {e}", flush=True)
    return L, model, sam, device, classes


def build_busi_dataset(L, model):
    import torchvision.transforms as T
    # CustomImageDataset from the user's BUSI code (embedded here to avoid cross-file import)
    from PIL import Image
    from torch.utils.data import Dataset

    class BusiSeg(Dataset):
        def __init__(self, root, transform, mask_transform):
            self.root, self.transform, self.mask_transform = root, transform, mask_transform
            self.class_to_idx = {d: i for i, d in enumerate(sorted(os.listdir(root)))}
            print(f"[busi] classes: {self.class_to_idx}", flush=True)
            self.samples = []
            for cname, label in self.class_to_idx.items():
                cdir = os.path.join(root, cname)
                if not os.path.isdir(cdir):
                    continue
                for fn in sorted(os.listdir(cdir)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")) and "_mask" not in fn:
                        ip = os.path.join(cdir, fn)
                        mp = os.path.join(cdir, fn.replace(".png", "_mask.png"))
                        if not os.path.exists(mp):
                            base = fn.rsplit(".", 1)[0]
                            cand = [f for f in os.listdir(cdir) if f.startswith(base) and "_mask" in f]
                            mp = os.path.join(cdir, cand[0]) if cand else None
                        self.samples.append((ip, label, mp))

        def __len__(self): return len(self.samples)

        def __getitem__(self, i):
            ip, label, mp = self.samples[i]
            image = Image.open(ip).convert("RGB")
            if mp and os.path.exists(mp):
                mask = Image.open(mp).convert("L")
            else:
                mask = Image.new("L", image.size, 0)
            img_t = self.transform(image)
            mask_t = (self.mask_transform(mask) > 0.5).float()
            return img_t, mask_t, ip, int(label)

    ds = BusiSeg(CONFIG["BUSI_DATA_DIR"], transform=model.preprocess,
                 mask_transform=T.Compose([T.ToTensor()]))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        CONFIG["MAX_SAMPLES"] = 6

    os.makedirs(os.path.dirname(CONFIG["OUT_CSV"]), exist_ok=True)
    L, model, sam, device, classes = load_pipeline()
    # BUSI-appropriate config: ultrasound lesions OVER-segment (benign precision ~0.20), the
    # opposite of Kvasir. The Kvasir growth/blob/CRF steps make it worse, so disable them here.
    for k, v in BUSI_OVERRIDES.items():
        L.REVISION_CONFIG[k] = v
    print(f"[busi cfg] {BUSI_OVERRIDES}", flush=True)
    ds = build_busi_dataset(L, model)

    normal_idx = ds.class_to_idx.get("normal", None)
    seg_idx = [i for i in range(len(ds))
               if not (CONFIG["SKIP_NORMAL"] and ds.samples[i][1] == normal_idx)]
    if CONFIG["MAX_SAMPLES"]:
        seg_idx = seg_idx[:CONFIG["MAX_SAMPLES"]]
    print(f"[busi] segmenting {len(seg_idx)} lesion images (skip_normal={CONFIG['SKIP_NORMAL']})", flush=True)

    # force per-image target class (annotation-free: image-level lesion label)
    L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = True

    import csv
    rows, dices, ious = [], [], []
    for k, idx in enumerate(seg_idx):
        try:
            img_t, mask_t, path, label = ds[idx]
            L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)   # <-- per-image lesion class
            sample = (img_t, mask_t, path)                          # Kvasir-style (image, gt, path)
            d, i = L._kvasir_segment_metrics_only(sample, model, sam, device,
                                                  CONFIG["XAI_WEIGHTS"],
                                                  fusion_strategy=CONFIG["FUSION_STRATEGY"],
                                                  use_iterative=CONFIG["USE_ITERATIVE"])
            dices.append(float(d)); ious.append(float(i))
            rows.append([os.path.basename(path), int(label), round(float(d), 4), round(float(i), 4)])
        except Exception as e:
            print(f"  [warn] idx {idx} failed: {e}", flush=True)
        if (k + 1) % 25 == 0:
            print(f"   {k+1}/{len(seg_idx)} done | running mean Dice {np.mean(dices):.4f}", flush=True)

    with open(CONFIG["OUT_CSV"], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "lesion_class", "dice", "iou"]); w.writerows(rows)

    print("\n" + "=" * 60)
    print("LENAS BUSI Segmentation Results")
    print("=" * 60)
    print(f"  Samples: {len(dices)}")
    if dices:
        d = np.array(dices); i = np.array(ious)
        print(f"  Mean Dice: {d.mean():.4f} +/- {d.std():.4f}")
        print(f"  Mean IoU : {i.mean():.4f} +/- {i.std():.4f}")
        print(f"  Dice==0  : {int((d < 1e-6).sum())} ({100*(d<1e-6).mean():.1f}%)")
        # per-class breakdown
        for cname, cidx in ds.class_to_idx.items():
            sub = [dd for dd, r in zip(dices, rows) if r[1] == cidx]
            if sub:
                print(f"    {cname:>10}: mean Dice {np.mean(sub):.4f}  (n={len(sub)})")
    print(f"  CSV -> {CONFIG['OUT_CSV']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
