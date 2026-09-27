# Hard-coded paths to set before running

Edit these lines to point to your local copies of the datasets and checkpoints. Nothing else needs to change.

| File | Line | Current value |
|---|---|---|
| `LENAS_Kvasir_revised.py` | 2652 | `SAM_CKPT_PATH = "/media/data/DARE/sam_vit_b_01ec64.pth"` |
| `LENAS_Kvasir_revised.py` | 2655 | `CAPSULE_DATA_PATH = "/home/satyajith/.cache/kagglehub/datasets/podakantisatyajith/capsule-vision-challenge-fin` |
| `LENAS_Kvasir_revised.py` | 2656 | `KVASIR_SEG_DATA_DIR = "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kv` |
| `alpha_sensitivity.py` | 32 | `"kvasir": "/media/data/DARE/kvasir-dataset/",` |
| `alpha_sensitivity.py` | 33 | `"busi":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",` |
| `kvasir_zero_dice_diagnostic.py` | 35 | `"KVASIR_IMAGES": "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-` |
| `kvasir_zero_dice_diagnostic.py` | 36 | `"KVASIR_MASKS":  "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-` |
| `kvasir_zero_dice_diagnostic.py` | 37 | `"SAM_CKPT":      "/media/data/DARE/sam_vit_b_01ec64.pth",` |
| `lenas_busi_run.py` | 34 | `"BUSI_DATA_DIR": "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",  # benign/ malignant/ normal/` |
| `lenas_busi_run.py` | 36 | `"SAM_CKPT": "/media/data/DARE/sam_vit_b_01ec64.pth",` |
| `lenas_figure_regen.py` | 32 | `"kvasir": "/media/data/DARE/best_kvasir_domain_classifier.pth",` |
| `lenas_figure_regen.py` | 33 | `"busi":   "/media/data/DARE/best_busi_domain_classifier.pth",` |
| `lenas_figure_regen.py` | 34 | `"capsule":"/media/data/DARE/best_capsule_domain_classifier.pth"})` |
| `lenas_figure_regen.py` | 42 | `"kvasir": "/media/data/DARE/kvasir-dataset/",` |
| `lenas_figure_regen.py` | 43 | `"busi":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",` |
| `lenas_figure_regen.py` | 44 | `"capsule":"/root/.cache/kagglehub/datasets/podakantisatyajith/capsule-vision-challenge-final-dataset-2024/vers` |
| `lenas_figure_regen.py` | 45 | `sam_ckpt: str = "/media/data/DARE/sam_vit_b_01ec64.pth"` |
| `lenas_revision_experiments.py` | 47 | `kvasir_dir: str = "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir` |
| `lenas_revision_experiments.py` | 48 | `sam_ckpt: str = "/media/data/DARE/sam_vit_b_01ec64.pth"` |
| `train_lenas_classifier.py` | 48 | `"POLYP_DIR":  "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG` |
| `train_lenas_classifier.py` | 195 | `"DATA_DIR": "/media/data/DARE/kvasir-dataset/",` |
| `train_lenas_classifier.py` | 198 | `"DATA_DIR": "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",` |
| `zero_shot_baselines.py` | 36 | `"images": "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/ima` |
| `zero_shot_baselines.py` | 37 | `"masks":  "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/mas` |
| `zero_shot_baselines.py` | 43 | `"root":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT",` |
| `zero_shot_baselines.py` | 157 | `SAM3_CKPT = "/media/data/DARE/sam3.pt"   # local gated checkpoint` |
| `zero_shot_baselines.py` | 232 | `BIOMEDPARSE_REPO = os.environ.get("BIOMEDPARSE_REPO", "/media/data/DARE/BiomedParse")  # cloned repo path` |
