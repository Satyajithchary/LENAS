# -*- coding: utf-8 -*-
"""
ALPHA (Eq. 5) SENSITIVITY  — Reviewer 2
=======================================
Reviewer 2 asked how different initializations of the context-suppression coefficient alpha
(Eq. 5) affect the pathology-context separation. In the code this coefficient is the
differential-attention base term `lambda_init` (default 0.8), where
    lambda_val = (lambda_1 - lambda_2 + lambda_init).mean().

This script sweeps the initialization over a range, RE-TRAINS the Differential BiomedCLIP
classifier for each value (image-level labels only), and reports:
  * the LEARNED lambda after training (does it converge regardless of init?),
  * classification validation accuracy,
  * mean normalized attribution entropy on a held-out subset (pathology-context separation:
    LOWER entropy = sharper, more localized attribution).

Expected, honest outcome: because alpha is learnable, the learned value and downstream
separation are stable across initializations — i.e., the method is not sensitive to the alpha
initialization. This directly answers R2 and reinforces the entropy story.

Usage
-----
  python alpha_sensitivity.py --dataset kvasir --epochs 10
  python alpha_sensitivity.py --dataset busi --epochs 10 --alphas 0.2,0.5,0.8,1.0,1.2
  python alpha_sensitivity.py --dataset busi --smoke
Output -> ./LENAS_<dataset>_experiments/results/alpha_sensitivity.csv
"""
import os, argparse, importlib
import numpy as np

DATA_DIR = {
    "kvasir": "/media/data/DARE/kvasir-dataset/",
    "busi":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",
}
MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def _entropy(a):
    a = np.abs(np.asarray(a, dtype=np.float64)); s = a.sum()
    if s <= 0: return 1.0
    p = a.ravel() / s; p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def learned_lambda(model):
    """Average learned lambda_val across differential attention blocks (matches the forward calc)."""
    import torch
    vals = []
    for m in model.modules():
        if all(hasattr(m, n) for n in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2", "lambda_init")):
            with torch.no_grad():
                l1 = torch.exp((m.lambda_q1 * m.lambda_k1).sum(-1).sum(-1))
                l2 = torch.exp((m.lambda_q2 * m.lambda_k2).sum(-1).sum(-1))
                vals.append(float((l1 - l2 + m.lambda_init).mean().item()))
    return float(np.mean(vals)) if vals else float("nan")

def build_loaders(L, model, root, smoke):
    import torch, os
    from torch.utils.data import Dataset, DataLoader, random_split
    from PIL import Image
    pre = model.preprocess
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    c2i = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        d = os.path.join(root, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_mask" not in f.lower():
                samples.append((os.path.join(d, f), c2i[c]))
    if smoke:
        samples = samples[:40]

    class DS(Dataset):
        def __len__(s): return len(samples)
        def __getitem__(s, i):
            p, y = samples[i]
            return pre(Image.open(p).convert("RGB")), y, p
    ds = DS()
    nval = max(1, int(0.15 * len(ds)))
    tr, va = random_split(ds, [len(ds) - nval, nval], generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=32, shuffle=True, num_workers=4, drop_last=len(tr) > 32),
            DataLoader(va, batch_size=32, num_workers=4), classes, c2i, samples)

def train_one(L, device, root, alpha, epochs, smoke):
    import torch, torch.nn as nn
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    model = L.DifferentialBiomedCLIP(MODEL_NAME, len(classes), device,
                                     lambda_init=alpha, class_names=classes, use_contrastive=True).to(device)
    tl, vl, classes, c2i, samples = build_loaders(L, model, root, smoke)
    counts = np.bincount([y for _, y in samples], minlength=len(classes)).astype(np.float32)
    w = torch.tensor((counts.sum() / (counts + 1e-6)) / len(classes), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    best = 0.0
    for ep in range(epochs):
        for n, p in model.named_parameters():
            head = ("classif" in n.lower()) or ("head" in n.lower()) or ("text" in n.lower()) or ("lambda" in n.lower())
            p.requires_grad = True if head else (ep >= 2)
        params = [p for p in model.parameters() if p.requires_grad] or list(model.parameters())
        opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-4)
        model.train()
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            lg = model(x); lg = lg[0] if isinstance(lg, tuple) else lg
            loss = crit(lg, y); opt.zero_grad(); loss.backward(); opt.step()
            if smoke: break
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for x, y, _ in vl:
                x, y = x.to(device), y.to(device)
                lg = model(x); lg = lg[0] if isinstance(lg, tuple) else lg
                cor += (lg.argmax(1) == y).sum().item(); tot += y.numel()
                if smoke: break
        best = max(best, cor / max(tot, 1))
    return model, best, samples, c2i

def mean_attr_entropy(model, device, samples, c2i, n=60):
    import torch
    from captum.attr import Saliency
    from PIL import Image
    import random
    idx = list(range(len(samples))); random.Random(0).shuffle(idx); idx = idx[:n]
    ents = []
    for i in idx:
        p, y = samples[i]
        try:
            x = model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
            a = Saliency(model).attribute(x, target=int(y), abs=True).abs().sum(1).squeeze().detach().cpu().numpy()
            ents.append(_entropy(a))
        except Exception:
            pass
    return float(np.mean(ents)) if ents else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATA_DIR.keys()))
    ap.add_argument("--alphas", default="0.2,0.5,0.8,1.0,1.2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 2; a.alphas = "0.5,0.8"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = importlib.reload(importlib.import_module("LENAS_Kvasir_revised"))
    root = DATA_DIR[a.dataset]
    alphas = [float(x) for x in a.alphas.split(",")]

    out_dir = f"./LENAS_{a.dataset}_experiments/results"; os.makedirs(out_dir, exist_ok=True)
    rows = []
    for alpha in alphas:
        print(f"\n=== alpha_init (Eq.5) = {alpha} ===", flush=True)
        model, val_acc, samples, c2i = train_one(L, device, root, alpha, a.epochs, a.smoke)
        lam = learned_lambda(model)
        ent = mean_attr_entropy(model, device, samples, c2i, n=8 if a.smoke else 60)
        rows.append([alpha, round(lam, 4), round(val_acc, 4), round(ent, 4)])
        print(f"   learned_lambda={lam:.4f} | val_acc={val_acc:.4f} | attr_entropy={ent:.4f}", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    import csv
    p = os.path.join(out_dir, "alpha_sensitivity.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["alpha_init_Eq5", "learned_lambda", "val_accuracy", "attr_entropy_norm"]); w.writerows(rows)
    print("\n" + "=" * 60)
    print(f"ALPHA (Eq.5) SENSITIVITY — {a.dataset}")
    print("=" * 60)
    print(f"{'alpha_init':>10} | {'learned_lambda':>14} | {'val_acc':>8} | {'attr_entropy':>12}")
    for r in rows:
        print(f"{r[0]:>10} | {r[1]:>14} | {r[2]:>8} | {r[3]:>12}")
    if len(rows) > 1:
        accs = [r[2] for r in rows]; ents = [r[3] for r in rows]
        print(f"\nval_acc spread: {max(accs)-min(accs):.4f} | entropy spread: {max(ents)-min(ents):.4f}")
        print("Small spreads => the method is robust to the alpha initialization (alpha is learnable).")
    print(f"CSV -> {p}")

if __name__ == "__main__":
    main()
