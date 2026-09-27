# -*- coding: utf-8 -*-
"""
BUSI DIAGNOSTIC  (per-stage, per-class)
=======================================
Reuses the Kvasir diagnostic's instrumentation (diagnose_one) on BUSI data with the per-image
lesion class as the XAI target. Reports, split by benign/malignant:
  - per-stage Dice / precision / recall (SAM_raw -> snake -> iterative -> final_CRF)
  - failure categories (no_overlap / tiny_overlap / ok) and GT-center-in-bbox
so we can see whether the low benign Dice is a LOCALISATION miss (bbox off the lesion) or a
COVERAGE problem (recall too low), and fix the right thing -- exactly as we did for Kvasir.

Usage:
  python busi_diagnostic.py --max 150
  python busi_diagnostic.py --smoke
  # optional knob overrides to test hypotheses (annotation-free):
  python busi_diagnostic.py --max 150 --no-suppress          # turn off endoscopy border/specular suppression
  python busi_diagnostic.py --max 150 --grow-pct 55 --dilate 35
"""
import os, argparse, importlib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-suppress", action="store_true", help="disable SUPPRESS_BORDER_SPECULAR (endoscopy-specific)")
    ap.add_argument("--grow-pct", type=int, default=None)
    ap.add_argument("--dilate", type=int, default=None)
    ap.add_argument("--no-growth", action="store_true", help="disable recall growth (BUSI over-segments)")
    ap.add_argument("--no-blob", action="store_true", help="disable blob-coverage candidate")
    ap.add_argument("--sharpen-pct", type=int, default=None, help="XAI_SHARPEN_PERCENTILE (higher=tighter prompts)")
    ap.add_argument("--npeaks", type=int, default=None, help="KVASIR_N_PEAKS candidates")
    ap.add_argument("--no-crf", action="store_true", help="disable CRF (hurts BUSI)")
    ap.add_argument("--no-candidate", action="store_true", help="disable reward candidate selection (BUSI over-selects)")
    ap.add_argument("--predicted-class", action="store_true", help="use classifier predicted class (like original BUSI), not GT label")
    ap.add_argument("--out", type=str, default="./LENAS_BUSI_outputs/results/busi_diagnostic.csv")
    args = ap.parse_args()
    if args.smoke:
        args.max = 8

    B = importlib.import_module("lenas_busi_run")
    D = importlib.import_module("kvasir_zero_dice_diagnostic")
    L, model, sam, device, classes = B.load_pipeline()
    ds = B.build_busi_dataset(L, model)

    # optional annotation-free knob overrides (to test the BUSI hypotheses)
    if args.no_suppress:
        L.REVISION_CONFIG["SUPPRESS_BORDER_SPECULAR"] = False
        print("[cfg] SUPPRESS_BORDER_SPECULAR = False")
    if args.grow_pct is not None:
        L.REVISION_CONFIG["KVASIR_GROW_PERCENTILE"] = args.grow_pct
        print(f"[cfg] KVASIR_GROW_PERCENTILE = {args.grow_pct}")
    if args.dilate is not None:
        L.REVISION_CONFIG["KVASIR_GROW_DILATE"] = args.dilate
        print(f"[cfg] KVASIR_GROW_DILATE = {args.dilate}")
    if args.no_growth:
        L.REVISION_CONFIG["KVASIR_RECALL_GROWTH"] = False; print("[cfg] KVASIR_RECALL_GROWTH = False")
    if args.no_blob:
        L.REVISION_CONFIG["KVASIR_BLOB_COVERAGE"] = False; print("[cfg] KVASIR_BLOB_COVERAGE = False")
    if args.sharpen_pct is not None:
        L.REVISION_CONFIG["XAI_SHARPEN_PERCENTILE"] = args.sharpen_pct; print(f"[cfg] XAI_SHARPEN_PERCENTILE = {args.sharpen_pct}")
    if args.npeaks is not None:
        L.REVISION_CONFIG["KVASIR_N_PEAKS"] = args.npeaks; print(f"[cfg] KVASIR_N_PEAKS = {args.npeaks}")
    if args.no_crf:
        L.REVISION_CONFIG["USE_CRF"] = False; print("[cfg] USE_CRF = False")
    if args.no_candidate:
        L.REVISION_CONFIG["KVASIR_CANDIDATE_SELECTION"] = False; print("[cfg] KVASIR_CANDIDATE_SELECTION = False")

    normal_idx = ds.class_to_idx.get("normal", None)
    lesion = [i for i in range(len(ds)) if ds.samples[i][1] != normal_idx]
    import random as _r; _r.Random(0).shuffle(lesion)   # mix benign+malignant
    idxs = lesion[:args.max]
    print(f"[busi-diag] analyzing {len(idxs)} lesion images", flush=True)

    L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = (not args.predicted_class)
    if args.predicted_class:
        print("[cfg] using classifier PREDICTED class (KVASIR_FORCE_POLYP_CLASS=False)")
    records = []
    for k, i in enumerate(idxs):
        img_t, mask_t, path, label = ds[i]
        L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)   # per-image lesion class
        try:
            rec = D.diagnose_one((img_t, mask_t, path), model, sam)
            rec["lesion_class"] = int(label)
            records.append(rec)
        except Exception as e:
            print(f"  [warn] idx {i}: {e}", flush=True)
        if (k + 1) % 25 == 0:
            print(f"   {k+1}/{len(idxs)} done", flush=True)

    if not records:
        print("No records."); return

    inv = {v: k for k, v in ds.class_to_idx.items()}

    def agg(subset, tag):
        if not subset:
            return
        dj = np.array([r["dice"] for r in subset])
        print(f"\n=== {tag}  (n={len(subset)}) ===")
        print(f"  mean Dice {dj.mean():.4f} | zeros {int((dj<1e-6).sum())} ({100*(dj<1e-6).mean():.0f}%)")
        for key, lab in [("dice_sam", "SAM_raw"), ("dice_snake", "snake"),
                         ("dice_iter", "iterative"), ("dice", "final_CRF")]:
            v = [r[key] for r in subset if r.get(key, -1) is not None and r.get(key, -1) >= 0]
            pk = key.replace("dice", "prec"); rk = key.replace("dice", "rec")
            ps = [r.get(pk, -1) for r in subset if r.get(pk, -1) >= 0]
            rs = [r.get(rk, -1) for r in subset if r.get(rk, -1) >= 0]
            if v:
                extra = f" | prec {np.mean(ps):.3f} recall {np.mean(rs):.3f}" if ps else ""
                print(f"    {lab:>10}: Dice {np.mean(v):.4f}{extra}")
        bh = [int(r.get("gt_center_in_bbox", 0)) for r in subset if r.get("gt_center_in_bbox", -1) >= 0]
        if bh:
            print(f"    GT-center-in-bbox: {100*np.mean(bh):.0f}%")
        from collections import Counter
        print(f"    categories: {dict(Counter(r['category'] for r in subset))}")

    agg(records, "ALL lesions")
    for cidx in sorted(set(r["lesion_class"] for r in records)):
        agg([r for r in records if r["lesion_class"] == cidx], f"class={inv.get(cidx, cidx)}")

    # write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    keys = ["lesion_class", "dice", "dice_sam", "dice_snake", "dice_iter",
            "prec_sam", "rec_sam", "prec_final", "rec_final",
            "gt_center_in_bbox", "category"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in records:
            w.writerow([r.get(k, "") for k in keys])
    print(f"\nCSV -> {args.out}")


if __name__ == "__main__":
    main()
