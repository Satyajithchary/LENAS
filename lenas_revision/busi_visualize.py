# -*- coding: utf-8 -*-
"""
BUSI QUALITATIVE VISUALS (visual proof)
=======================================
Generates the pipeline's multi-panel figures (Original / GT / Saliency / Fused+Prompts /
SAM / Snake / Final / Error) for representative BUSI images, using the locked BUSI config and
the per-image lesion class as the XAI target.

Usage:
  python busi_visualize.py                 # ~6 figures (best benign/malignant + typical)
  python busi_visualize.py --n-scan 80 --per-class 3
Figures are saved under ./LENAS_BUSI_outputs/figures/.
"""
import os, argparse, importlib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scan", type=int, default=60, help="images to score before picking examples")
    ap.add_argument("--per-class", type=int, default=3, help="best examples per class to render")
    ap.add_argument("--out", type=str, default="./LENAS_BUSI_outputs")
    a = ap.parse_args()

    B = importlib.import_module("lenas_busi_run")
    L, model, sam, device, classes = B.load_pipeline()
    for k, v in B.BUSI_OVERRIDES.items():
        L.REVISION_CONFIG[k] = v
    L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = True
    L.REVISION_CONFIG["SAVE_FIGURES"] = True
    L.REVISION_CONFIG["OUTPUT_DIR"] = a.out
    if hasattr(L, "setup_output_dirs"):
        try: L.setup_output_dirs()
        except Exception as e: print(f"[warn] setup_output_dirs: {e}")

    ds = B.build_busi_dataset(L, model)
    inv = {v: k for k, v in ds.class_to_idx.items()}
    normal_idx = ds.class_to_idx.get("normal", None)
    lesion = [i for i in range(len(ds)) if ds.samples[i][1] != normal_idx]
    import random
    random.Random(0).shuffle(lesion)
    scan = lesion[:a.n_scan]

    # 1) score a scan subset to find good examples
    scored = []
    for k, i in enumerate(scan):
        img_t, mask_t, path, label = ds[i]
        L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)
        try:
            d, _ = L._kvasir_segment_metrics_only((img_t, mask_t, path), model, sam, device,
                                                  B.CONFIG["XAI_WEIGHTS"], fusion_strategy="weighted_average", use_iterative=True)
            scored.append((i, int(label), float(d)))
        except Exception as e:
            print(f"  [warn] score {i}: {e}")
        if (k + 1) % 20 == 0:
            print(f"   scored {k+1}/{len(scan)}", flush=True)

    # 2) pick best per class + one median per class
    chosen = []
    for cidx in sorted(set(s[1] for s in scored)):
        cls = sorted([s for s in scored if s[1] == cidx], key=lambda s: -s[2])
        chosen += cls[:a.per_class]                       # best
        if len(cls) > a.per_class:
            chosen.append(cls[len(cls) // 2])             # one typical/median
    print(f"[viz] rendering {len(chosen)} figures", flush=True)

    # 3) render the multi-panel figure for each
    import matplotlib
    matplotlib.use("Agg")
    for (i, label, d) in chosen:
        img_t, mask_t, path, _ = ds[i]
        L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)
        try:
            L.visualize_kvasir_segmentation_pipeline((img_t, mask_t, path), model, sam, device,
                                                     B.CONFIG["XAI_WEIGHTS"], fusion_strategy="weighted_average")
            # rename the just-saved figure to something descriptive if possible
            print(f"   rendered {inv.get(label,label)} Dice={d:.3f}: {os.path.basename(path)}", flush=True)
        except Exception as e:
            print(f"  [warn] render {i}: {e}")

    figdir = os.path.join(a.out, "figures")
    print(f"\nFigures saved under: {figdir}")
    try:
        print("Files:", sorted(os.listdir(figdir))[-len(chosen):])
    except Exception:
        pass

if __name__ == "__main__":
    main()
