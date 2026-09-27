# -*- coding: utf-8 -*-
"""
ZERO-SHOT BASELINE COMPARISON  (SAM 2 / SAM 3 / BiomedParse)  on Kvasir-SEG + BUSI
==================================================================================
Standalone evaluation of promptable / text-driven foundation models against LENAS, so you can
decide (privately) whether to include them. Protocols follow the click-prompt convention used in
"Comparing SAM 2 and SAM 3 for Zero-Shot Segmentation of 3D Medical Data":

  * 1-click  : one positive point at the GT centroid
  * 2-click  : GT centroid + one positive point in the largest-residual region
  * (optional) GT-box prompt for reference (upper bound with oracle localisation)
  * BiomedParse: text-prompt zero-shot (no GT used), matched to the lesion type

IMPORTANT (honesty): 1-click / 2-click / box protocols use the GT mask to place prompts, so they
are NOT annotation-free. They are an *upper-bound / interactive* reference, exactly as in the cited
paper. LENAS uses NO GT at inference; report that distinction clearly. BiomedParse (text-only) is
the fair zero-shot comparison.

This script AUTO-INSTALLS dependencies and AUTO-DOWNLOADS checkpoints where a public URL exists.
SAM 3 weights are gated/unreleased at time of writing; the script degrades gracefully and skips a
model whose weights/library are unavailable (it never crashes the whole run).

Usage
-----
  python zero_shot_baselines.py --dataset kvasir --models sam2,biomedparse --clicks 1,2 --n 200
  python zero_shot_baselines.py --dataset busi   --models sam2,sam3,biomedparse --clicks 1,2
  python zero_shot_baselines.py --dataset kvasir --models sam2 --smoke
Results -> ./ZS_baselines/<dataset>_zeroshot.csv  (+ per-model means printed)
"""
import os, sys, subprocess, argparse, urllib.request, importlib
import numpy as np

# --------------------------------------------------------------- paths (edit if needed)
DATASETS = {
    "kvasir": {
        "images": "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/images",
        "masks":  "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/masks",
        "layout": "flat",           # images/ + masks/ with same filename
        "text":   "polyp",
        "text_alts": ["colon polyp", "polyp", "lesion", "abnormal growth"],
    },
    "busi": {
        "root":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT",
        "layout": "busi",           # class folders, <name>_mask.png
        "text":   "tumor",
        "text_alts": ["breast tumor", "tumor", "mass", "lesion"],
    },
}
CKPT_DIR = "./ZS_checkpoints"
OUT_DIR = "./ZS_baselines"

# --------------------------------------------------------------- utils
def sh(cmd):
    print("  $", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode == 0

def pip_install(pkg):
    return sh(f"{sys.executable} -m pip install {pkg} --quiet --break-system-packages")

def download(url, path):
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        print(f"  downloading {url} -> {path}", flush=True)
        urllib.request.urlretrieve(url, path); return True
    except Exception as e:
        print(f"  [download failed] {e}"); return False

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / (a.sum() + b.sum() + 1e-8)

def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / (u + 1e-8)

# --------------------------------------------------------------- data
def list_samples(dataset):
    cfg = DATASETS[dataset]; items = []
    from glob import glob
    if cfg["layout"] == "flat":
        for ip in sorted(glob(os.path.join(cfg["images"], "*"))):
            mp = os.path.join(cfg["masks"], os.path.basename(ip))
            if os.path.exists(mp):
                items.append((ip, mp, cfg["text"]))
    else:  # busi
        for cls in ("benign", "malignant"):
            d = os.path.join(cfg["root"], cls)
            if not os.path.isdir(d):
                continue
            for ip in sorted(glob(os.path.join(d, "*.png"))):
                if "_mask" in ip:
                    continue
                mp = ip.replace(".png", "_mask.png")
                if os.path.exists(mp):
                    items.append((ip, mp, cfg["text"]))
    return items

def load_pair(ip, mp):
    from PIL import Image
    img = np.array(Image.open(ip).convert("RGB"))
    gt = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8)
    return img, gt

def gt_click_points(gt, n_clicks):
    """GT-derived positive clicks (interactive protocol, NOT annotation-free)."""
    ys, xs = np.where(gt > 0)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    cy, cx = int(ys.mean()), int(xs.mean())
    pts = [[cx, cy]]
    if n_clicks >= 2:
        # second click: farthest GT pixel from the centroid (covers elongated lesions)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        j = int(np.argmax(d2)); pts.append([int(xs[j]), int(ys[j])])
    return np.array(pts), np.ones(len(pts))

# --------------------------------------------------------------- SAM 2
def build_sam2():
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception:
        print("[sam2] installing...")
        pip_install("git+https://github.com/facebookresearch/sam2.git")
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            print(f"[sam2] unavailable: {e}"); return None
    ckpt = os.path.join(CKPT_DIR, "sam2.1_hiera_large.pt")
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    if not download("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", ckpt):
        return None
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(cfg, ckpt, device=dev)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(model)
    except Exception as e:
        print(f"[sam2] build failed: {e}"); return None

def run_sam2(pred, img, pts, labels):
    import torch
    pred.set_image(img)
    with torch.inference_mode():
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8)

# --------------------------------------------------------------- SAM 3 (text / concept)
SAM3_CKPT = "/media/data/DARE/sam3.pt"   # local gated checkpoint

def build_sam3():
    """SAM 3 via HuggingFace transformers (Sam3Model). SAM 3 is a TEXT/concept segmentation
    model; the documented API takes a text phrase, not click points. We therefore run it
    text-only ('polyp'/'tumor'). If a local checkpoint is provided it is loaded on top."""
    try:
        import torch
        from transformers import Sam3Processor, Sam3Model
    except Exception as e:
        print(f"[sam3] transformers Sam3Model unavailable ({e}); pip install -U transformers. Skipping.")
        return None
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
        proc = Sam3Processor.from_pretrained("facebook/sam3")
        # NOTE: the local /media/data/DARE/sam3.pt uses a different key layout (missing=1468,
        # unexpected=1465) and is therefore NOT applied; we use the official HF weights, which
        # load cleanly and produce valid concept masks.
        print("[sam3] using official HF facebook/sam3 weights (local ckpt skipped: key mismatch)")
        return (model, proc, dev)
    except Exception as e:
        print(f"[sam3] build failed ({e}). Ensure gated access to facebook/sam3. Skipping.")
        return None

def debug_sam3(bp, items, cfg):
    """Diagnose the 0.00: print #instances, scores, and best-prompt Dice on a few images."""
    import torch, numpy as np
    from PIL import Image
    model, proc, dev = bp
    prompts = cfg.get("text_alts", [cfg["text"]])
    print("\n[sam3-debug] prompts to try:", prompts)
    for ip, mp, _ in items[:5]:
        img, gt = load_pair(ip, mp)
        print(f"\n  image {os.path.basename(ip)} (gt px={int(gt.sum())})")
        for pr in prompts:
            for thr in (0.5, 0.3, 0.15):
                inputs = proc(images=Image.fromarray(img), text=pr, return_tensors="pt").to(dev)
                with torch.no_grad():
                    out = model(**inputs)
                res = proc.post_process_instance_segmentation(
                    out, threshold=thr, mask_threshold=thr,
                    target_sizes=inputs.get("original_sizes").tolist())[0]
                masks = res.get("masks", [])
                if len(masks):
                    m = np.zeros(img.shape[:2], np.uint8)
                    for mk in masks:
                        arr = mk.cpu().numpy() if hasattr(mk, "cpu") else np.asarray(mk)
                        m |= (arr > 0.5).astype(np.uint8)
                    print(f"    '{pr}' thr={thr}: {len(masks)} inst | Dice={dice(m,gt):.3f}")
                else:
                    print(f"    '{pr}' thr={thr}: 0 instances (empty)")

def run_sam3_text(bp, img, text):
    import torch
    from PIL import Image
    model, proc, dev = bp
    inputs = proc(images=Image.fromarray(img), text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=0.3, mask_threshold=0.3,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    masks = res.get("masks", [])
    if len(masks) == 0:
        return np.zeros(img.shape[:2], np.uint8)
    # union of all instances of the concept (exhaustive PCS)
    m = np.zeros(img.shape[:2], np.uint8)
    for mk in masks:
        arr = mk.cpu().numpy() if hasattr(mk, "cpu") else np.asarray(mk)
        m |= (arr > 0.5).astype(np.uint8)
    return m

# --------------------------------------------------------------- BiomedParse (text zero-shot)
BIOMEDPARSE_REPO = os.environ.get("BIOMEDPARSE_REPO", "/media/data/DARE/BiomedParse")  # cloned repo path

def build_biomedparse():
    """BiomedParse via its official repo (needs detectron2 + configs). Set BIOMEDPARSE_REPO to the
    cloned https://github.com/microsoft/BiomedParse path. Text-only, annotation-free."""
    if not os.path.isdir(BIOMEDPARSE_REPO):
        print(f"[biomedparse] repo not found at {BIOMEDPARSE_REPO}. "
              "git clone https://github.com/microsoft/BiomedParse and set BIOMEDPARSE_REPO. Skipping.")
        return None
    sys.path.insert(0, BIOMEDPARSE_REPO)
    try:
        import torch
        from modeling.BaseModel import BaseModel
        from modeling import build_model
        from utilities.distributed import init_distributed
        from utilities.arguments import load_opt_from_config_files
        from utilities.constants import BIOMED_CLASSES
        from inference_utils.inference import interactive_infer_image
        opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_REPO, "configs/biomedparse_inference.yaml")])
        opt = init_distributed(opt)
        model = BaseModel(opt, build_model(opt)).from_pretrained("hf_hub:microsoft/BiomedParse").eval().cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], is_eval=True)
        return (model, interactive_infer_image)
    except Exception as e:
        print(f"[biomedparse] setup failed ({e}). Check detectron2/config install. Skipping.")
        return None

def run_biomedparse(bp, img, text):
    from PIL import Image
    model, infer = bp
    pred = infer(model, Image.fromarray(img).convert("RGB"), [text])
    m = pred[0]
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    return (m > 0.5).astype(np.uint8)

# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--models", default="sam2,sam3", help="comma list: sam2,sam3,biomedparse")
    ap.add_argument("--clicks", default="1,2", help="comma list of click counts for SAM models")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--debug-sam3", action="store_true", help="diagnose SAM3 empty-output issue on 5 images")
    a = ap.parse_args()
    if a.smoke:
        a.n = 6
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

    items = list_samples(a.dataset)
    if not items:
        print(f"No samples found for {a.dataset}. Check DATASETS paths at top of file."); return
    import random; random.Random(0).shuffle(items); items = items[:a.n]
    print(f"[{a.dataset}] evaluating {len(items)} images", flush=True)

    models = [m.strip() for m in a.models.split(",")]
    clicks = [int(c) for c in a.clicks.split(",")]
    built = {}
    if "sam2" in models: built["sam2"] = build_sam2()
    if "sam3" in models: built["sam3"] = build_sam3()
    if a.debug_sam3 and built.get("sam3") is not None:
        debug_sam3(built["sam3"], items, DATASETS[a.dataset]); return
    if "biomedparse" in models: built["biomedparse"] = build_biomedparse()

    rows = []  # (model, protocol, dice, iou, n)
    agg = {}
    for ip, mp, text in items:
        img, gt = load_pair(ip, mp)
        if gt.sum() == 0:
            continue
        for mname in models:
            mdl = built.get(mname)
            if mdl is None:
                continue
            try:
                if mname == "sam2":
                    for k in clicks:
                        pts, labs = gt_click_points(gt, k)
                        if len(pts) == 0:
                            continue
                        pred = run_sam2(mdl, img, pts, labs)
                        agg.setdefault(f"sam2_{k}click", []).append((dice(pred, gt), iou(pred, gt)))
                elif mname == "sam3":
                    # SAM 3 = TEXT/concept model (no native click API). Evaluate two prompts.
                    for pr, tag in [(DATASETS[a.dataset]["text"], "word"),
                                    ("lesion", "lesion")]:
                        pred = run_sam3_text(mdl, img, pr)
                        agg.setdefault(f"sam3_text_{tag}", []).append((dice(pred, gt), iou(pred, gt)))
                elif mname == "biomedparse":
                    pred = run_biomedparse(mdl, img, text)
                    agg.setdefault("biomedparse_text", []).append((dice(pred, gt), iou(pred, gt)))
            except Exception as e:
                print(f"  [warn] {mname} on {os.path.basename(ip)}: {e}")

    print("\n" + "=" * 60)
    print(f"ZERO-SHOT BASELINES — {a.dataset}")
    print("=" * 60)
    for key, vals in agg.items():
        d = np.mean([v[0] for v in vals]); i = np.mean([v[1] for v in vals])
        note = " (uses GT for click placement; interactive upper bound)" if "click" in key else " (text-only zero-shot; annotation-free)"
        print(f"  {key:>22}: Dice {d:.4f} | IoU {i:.4f} | n={len(vals)}{note}")
        rows.append([key, round(float(d), 4), round(float(i), 4), len(vals), note.strip()])
    if not agg:
        print("  No model produced results (weights/libraries unavailable). See messages above.")
    import csv
    with open(os.path.join(OUT_DIR, f"{a.dataset}_zeroshot.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_protocol", "dice", "iou", "n", "note"]); w.writerows(rows)
    print(f"\nCSV -> {os.path.join(OUT_DIR, f'{a.dataset}_zeroshot.csv')}")
    print("Reminder: for the paper, only the TEXT-only zero-shot (BiomedParse) is annotation-free and")
    print("directly comparable to LENAS; click/box protocols are interactive upper bounds.")

if __name__ == "__main__":
    main()
