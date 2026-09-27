# `lenas_revision/`: code that reproduces the results reported in the article

This folder contains the exact scripts that produced the segmentation, ablation, comparison, sensitivity, and supplementary results of the Communications AI & Computing article. They include the diagnosed corrections described in the article's revision (iterative refinement of the incoming mask, confidence-reward prompt selection, saliency-guided recall growth, and per-backbone preprocessing).

The modular package in the repository root (`models/`, `xai/`, `segmentation/`, ...) is the original implementation and is kept for reference.

**All scripts must be run from inside this folder.** They import `LENAS_Kvasir_revised.py` by module name.

## 1. Set the paths

Edit the lines listed in [`PATHS_TO_SET.md`](PATHS_TO_SET.md): dataset folders, classifier checkpoints, and the SAM checkpoint `sam_vit_b_01ec64.pth`.

## 2. Which script produces which result

| Result in the article | Script | Command (from this folder) |
|---|---|---|
| Classifier checkpoints (image-level labels only) | `train_lenas_classifier.py` | `python train_lenas_classifier.py --dataset kvasir` and `python train_lenas_classifier.py --dataset busi` |
| Main Table 2, LENAAFMIS on Kvasir-SEG (Dice 0.5981, full set) | `LENAS_Kvasir_revised.py` | `python LENAS_Kvasir_revised.py` |
| Main Table 2, LENAAFMIS on BUSI (Dice 0.3049) | `lenas_busi_run.py` | `python lenas_busi_run.py` (add `--smoke` for a quick test) |
| Main Table 2 U-Net baselines; Tables 3, 4 (CLIPSeg, SAM automatic), and 5; CRF before/after; Suppl. Table C10 (Kvasir-SEG) | `lenas_revision_experiments.py` | `python lenas_revision_experiments.py` (`--only <experiment>`, `--smoke`, `--force`) |
| Same experiments on BUSI | `lenas_busi_experiments.py` | `python lenas_busi_experiments.py` |
| Main Table 4: SAM 2 (1 and 2 clicks) and SAM 3 (text) | `zero_shot_baselines.py` | `python zero_shot_baselines.py --dataset kvasir --models sam2,sam3 --clicks 1,2` (repeat with `--dataset busi`) |
| Suppl. Table C9, initialisation of the context-suppression coefficient | `alpha_sensitivity.py` | `python alpha_sensitivity.py --dataset kvasir` (repeat with `--dataset busi`) |
| Qualitative figures (viridis, labelled colour bars) | `lenas_figure_regen.py` | see the header of the script |
| Diagnostics used during the revision (per-stage Dice, per-class BUSI) | `kvasir_zero_dice_diagnostic.py`, `busi_diagnostic.py`, `busi_visualize.py` | optional |

Every experiment script accepts `--smoke` for a fast end-to-end check before a full run. Results are written as CSV and Excel files. The numerical source data of all tables are also in Supplementary Data 1 of the article.

## 3. Notes on reproducibility

- Hardware: a single NVIDIA RTX A6000 Ada (47 GB); about 634 ms per image for the full pipeline.
- Preprocessing: the native BiomedCLIP preprocessing only. No data augmentation, resampling, or synthetic data is used.
- Segmentation masks are used only for evaluation, never for training or prompting (except the SAM 2 click baselines, which are reported as an interactive upper bound).
- Configuration is dataset-dependent, as described in the article: the reward candidate selection, blob coverage, and recall-growth stages are enabled for Kvasir-SEG and disabled for BUSI (`BUSI_OVERRIDES` in `lenas_busi_run.py`).
