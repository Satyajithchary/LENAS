# -*- coding: utf-8 -*-
"""
TRAIN AN IN-DOMAIN LENAS CLASSIFIER  (image-level labels ONLY -- no segmentation masks)
=======================================================================================
Why this exists
---------------
The Kvasir Dice/IoU were capped not by the prompting but by the ATTRIBUTION maps: the
capsule-endoscopy classifier is out-of-domain on colonoscopy, so its gradients are
diffuse noise and the prompts are seeded from nothing. The principled, annotation-free
fix (the lever you offered) is to train the SAME Differential BiomedCLIP backbone on a
colonoscopy/GI *classification* dataset using only image-level labels. The resulting
checkpoint is a drop-in replacement for the capsule classifier in LENAS_Kvasir_revised.py.

This uses ZERO segmentation masks, so LENAS remains annotation-free for segmentation.

Two ways to supply data (pick one in CONFIG):
  A) IMAGEFOLDER  : DATA_DIR/<class_name>/*.jpg  (e.g. HyperKvasir labeled-images:
                    'polyps', 'normal-cecum', 'normal-pylorus', 'ulcerative-colitis', ...)
  B) BINARY       : POLYP_DIR (all polyp frames, e.g. Kvasir-SEG/images) +
                    NORMAL_DIR (any normal colonoscopy frames). Uses only the folder label.

Then in
LENAS_Kvasir_revised.py main() set:
    MODEL_NAME stays the same
    CAPSULE_NUM_CLASSES  = <printed num_classes>
    CAPSULE_CLASS_LABELS = <printed class list, in index order>
    CAPSULE_MODEL_SAVE_PATH = "<SAVE_PATH below>"
and in REVISION_CONFIG set:
    "KVASIR_FORCE_POLYP_CLASS": True
    "KVASIR_TARGET_CLASS": <printed polyp index>
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ----------------------------------------------------------------------------- CONFIG
CONFIG = {
    "MODULE_NAME": "LENAS_Kvasir_revised",   # to import the identical DifferentialBiomedCLIP
    "MODEL_NAME":  "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",

    "MODE": "imagefolder",                   # "imagefolder" or "binary"
    # --- MODE A: imagefolder ---
    "DATA_DIR": "/path/to/colonoscopy_classification",   # subfolder per class
    # --- MODE B: binary polyp vs normal ---
    "POLYP_DIR":  "/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG/images",
    "NORMAL_DIR": "/path/to/normal_colonoscopy_frames",

    "SAVE_PATH": "best_kvasir_domain_classifier.pth",
    "EPOCHS": 15,
    "BATCH": 32,
    "LR": 3e-5,
    "VAL_FRAC": 0.15,
    "FREEZE_BACKBONE_EPOCHS": 2,             # train head first, then unfreeze
    "NUM_WORKERS": 4,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}
random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])

L = __import__(CONFIG["MODULE_NAME"])
DifferentialBiomedCLIP = L.DifferentialBiomedCLIP

# ----------------------------------------------------------------------------- dataset
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and "_mask" not in f.lower()]

def build_index():
    """Return (samples, class_names) where samples=[(path, label_idx), ...]."""
    if CONFIG["MODE"] == "imagefolder":
        root = CONFIG["DATA_DIR"]
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        samples = []
        for i, c in enumerate(classes):
            for p in list_images(os.path.join(root, c)):
                samples.append((p, i))
        return samples, classes
    else:
        classes = ["normal", "polyp"]          # index 0 = normal, 1 = polyp
        samples = [(p, 1) for p in list_images(CONFIG["POLYP_DIR"])]
        samples += [(p, 0) for p in list_images(CONFIG["NORMAL_DIR"])]
        return samples, classes

class ClsDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), y

# ----------------------------------------------------------------------------- train
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    per_class_correct, per_class_total = {}, {}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            pred = logits.argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
            for yi, pi in zip(y.cpu().numpy(), pred.cpu().numpy()):
                per_class_total[yi] = per_class_total.get(yi, 0) + 1
                per_class_correct[yi] = per_class_correct.get(yi, 0) + int(yi == pi)
    acc = correct / max(total, 1)
    return acc, per_class_correct, per_class_total

def main():
    device = CONFIG["DEVICE"]
    samples, classes = build_index()
    if len(samples) == 0:
        print("No images found. Check CONFIG paths / MODE."); return
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # Prefer the exact 'polyps' class over substring matches like 'dyed-lifted-polyps'
    def _find_polyp_idx(c2i):
        for exact in ("polyps", "polyp"):
            if exact in c2i:
                return c2i[exact]
        plain = [i for c, i in c2i.items() if "polyp" in c.lower() and "dyed" not in c.lower()]
        if plain:
            return plain[0]
        anyp = [i for c, i in c2i.items() if "polyp" in c.lower()]
        return anyp[0] if anyp else None
    polyp_idx = _find_polyp_idx(class_to_idx)
    print(f"Found {len(samples)} images across {num_classes} classes: {class_to_idx}")
    print(f"Polyp class index = {polyp_idx}")

    model = DifferentialBiomedCLIP(CONFIG["MODEL_NAME"], num_classes, device,
                                   class_names=classes, use_contrastive=True).to(device)
    preprocess = model.preprocess

    ds = ClsDataset(samples, preprocess)
    n_val = max(1, int(len(ds) * CONFIG["VAL_FRAC"]))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(CONFIG["SEED"]))
    tl = DataLoader(tr, batch_size=CONFIG["BATCH"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], drop_last=False)
    vl = DataLoader(va, batch_size=CONFIG["BATCH"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])

    # class-balanced CE (datasets like Kvasir-SEG-as-polyp can be imbalanced)
    counts = np.bincount([y for _, y in samples], minlength=num_classes).astype(np.float32)
    weights = torch.tensor((counts.sum() / (counts + 1e-6)) / num_classes, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    def set_backbone_grad(flag):
        for n, p in model.named_parameters():
            if "classifier" not in n and "head" not in n and "text" not in n:
                p.requires_grad = flag

    best_acc = 0.0
    for epoch in range(CONFIG["EPOCHS"]):
        set_backbone_grad(epoch >= CONFIG["FREEZE_BACKBONE_EPOCHS"])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=CONFIG["LR"], weight_decay=1e-4)
        model.train(); run = 0.0
        for x, y in tl:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        acc, pcc, pct = evaluate(model, vl, device)
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | loss {run/max(len(tl),1):.4f} | val acc {acc*100:.2f}%")
        if polyp_idx is not None and polyp_idx in pct:
            print(f"   polyp recall: {pcc.get(polyp_idx,0)}/{pct[polyp_idx]} "
                  f"= {100*pcc.get(polyp_idx,0)/max(pct[polyp_idx],1):.1f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(),
                        "class_to_idx": class_to_idx, "classes": classes,
                        "num_classes": num_classes, "polyp_idx": polyp_idx,
                        "val_acc": acc}, CONFIG["SAVE_PATH"])
            print(f"   ✅ saved best -> {CONFIG['SAVE_PATH']}")

    print("\n" + "=" * 70)
    print("DONE. Now update LENAS_Kvasir_revised.py main():")
    print(f"   CAPSULE_NUM_CLASSES  = {num_classes}")
    print(f"   CAPSULE_CLASS_LABELS = {classes}")
    print(f"   CAPSULE_MODEL_SAVE_PATH = '{CONFIG['SAVE_PATH']}'")
    print("and in REVISION_CONFIG:")
    print(f"   'KVASIR_FORCE_POLYP_CLASS': True,")
    print(f"   'KVASIR_TARGET_CLASS': {polyp_idx},")
    print("=" * 70)

DATASET_PRESETS = {
    "kvasir": {"MODE": "imagefolder",
               "DATA_DIR": "./kvasir-dataset/",
               "SAVE_PATH": "best_kvasir_domain_classifier.pth"},
    "busi":   {"MODE": "imagefolder",
               "DATA_DIR": "./BUSI_Dataset/Dataset_BUSI_with_GT/",
               "SAVE_PATH": "best_busi_domain_classifier.pth"},
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), default=None,
                    help="kvasir or busi -> sets DATA_DIR + SAVE_PATH automatically")
    args = ap.parse_args()
    if args.dataset:
        CONFIG.update(DATASET_PRESETS[args.dataset])
        print(f"[preset] dataset={args.dataset} -> DATA_DIR={CONFIG['DATA_DIR']} SAVE_PATH={CONFIG['SAVE_PATH']}")
    main()
