# -*- coding: utf-8 -*-
"""
LENAS BUSI REVISION EXPERIMENTS
===============================
BUSI twin of lenas_revision_experiments.py. Runs every reviewer-requested experiment on the
BUSI breast-ultrasound dataset, PER CLASS (benign / malignant), using the locked BUSI config
(plain pipeline: candidate-selection / growth / blob / CRF OFF, per-image GT lesion label).

Reuses lenas_busi_run.py for dataset + model loading, and the debugged pipeline in
LENAS_Kvasir_revised.py for segmentation. Resumable; --smoke for a quick end-to-end test;
exports source-data CSVs + a combined Excel workbook.

Experiments: table4_baselines (U-Net 1/5/10/100%), variantD (standard vs differential),
sensitivity (fusion + XAI weights + sharpen), recent_methods (CLIPSeg / SAM-auto), variability
(>=3 seeds), crf_ablation (before/after). All report per-class where meaningful.

Usage
-----
  python lenas_busi_experiments.py --smoke
  python -u lenas_busi_experiments.py 2>&1 | tee busi_experiments.log
  python lenas_busi_experiments.py --only sensitivity,crf_ablation
  python lenas_busi_experiments.py --force
"""
import os, sys, json, time, argparse, importlib
import numpy as np

OUT_DIR = "./LENAS_BUSI_experiments"
SEEDS = [0, 1, 2]
N_LENAS = 200          # subset for LENAS-based experiments (None = all lesions)
N_VARIANTD = 120
UNET_FRACTIONS = [0.01, 0.05, 0.10, 1.0]
UNET_IMG, UNET_EPOCHS, UNET_BS, UNET_TESTFRAC = 128, 40, 8, 0.2
XAI_WEIGHTS = {"Saliency": 0.279, "IntegratedGradients": 0.352, "GradientShap": 0.369}

# fill from recent BUSI papers (clearly labelled "reported in")
REPORTED_NUMBERS = {
    # "MethodX (reported in [ref])": {"busi_dice": None, "annotation": "zero-shot", "source": "[ref]"},
}

def log(*a): print(*a, flush=True)
def rp(name): return os.path.join(OUT_DIR, "results", f"{name}.csv")
def stp(): return os.path.join(OUT_DIR, "state.json")
def load_state():
    try: return json.load(open(stp()))
    except Exception: return {"done": []}
def mark_done(n):
    st = load_state()
    if n not in st["done"]: st["done"].append(n)
    json.dump(st, open(stp(), "w"), indent=2)
def write_csv(name, rows, header):
    import csv
    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    with open(rp(name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    log(f"   -> wrote {rp(name)}")

_P = {}
def pipe():
    if _P: return _P
    B = importlib.import_module("lenas_busi_run")
    L, model, sam, device, classes = B.load_pipeline()
    for k, v in B.BUSI_OVERRIDES.items():      # apply the locked BUSI plain-pipeline config
        L.REVISION_CONFIG[k] = v
    L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = True   # per-image GT label as target
    ds = B.build_busi_dataset(L, model)
    _P.update(dict(B=B, L=L, model=model, sam=sam, device=device, classes=classes, ds=ds))
    log(f"[busi cfg] {B.BUSI_OVERRIDES}")
    return _P

def lesion_indices(ds, n=None, seed=0):
    normal_idx = ds.class_to_idx.get("normal", None)
    idx = [i for i in range(len(ds)) if ds.samples[i][1] != normal_idx]
    import random; random.Random(seed).shuffle(idx)
    return idx[: (n or len(idx))]

def run_busi_subset(idxs, xai_weights=None, fusion="weighted_average", use_iter=True, overrides=None):
    P = pipe(); L, model, sam, device, ds = P["L"], P["model"], P["sam"], P["device"], P["ds"]
    xw = xai_weights or XAI_WEIGHTS
    saved = {}
    if overrides:
        for k, v in overrides.items():
            saved[k] = L.REVISION_CONFIG.get(k); L.REVISION_CONFIG[k] = v
    out = []  # (dice, iou, class)
    try:
        for j, i in enumerate(idxs):
            try:
                img_t, mask_t, path, label = ds[i]
                L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)
                d, iou = L._kvasir_segment_metrics_only((img_t, mask_t, path), model, sam, device,
                                                        xw, fusion_strategy=fusion, use_iterative=use_iter)
                out.append((float(d), float(iou), int(label)))
            except Exception as e:
                log(f"   [warn] idx {i}: {e}")
            if (j + 1) % 25 == 0:
                log(f"     {j+1}/{len(idxs)} done")
    finally:
        if overrides:
            for k, v in saved.items(): L.REVISION_CONFIG[k] = v
    return out

def by_class(out, ds):
    inv = {v: k for k, v in ds.class_to_idx.items()}
    res = {}
    for cidx in sorted(set(o[2] for o in out)):
        sub = [o[0] for o in out if o[2] == cidx]
        res[inv.get(cidx, cidx)] = (float(np.mean(sub)), len(sub))
    allm = float(np.mean([o[0] for o in out])) if out else 0.0
    return allm, res

# =============================================================== EXP1 U-Net baselines
def exp_table4_baselines():
    import torch, torch.nn as nn
    from PIL import Image
    import torchvision.transforms as T
    P = pipe(); ds, device = P["ds"], P["device"]
    # BUSI lesion images+masks (skip normal)
    normal_idx = ds.class_to_idx.get("normal", None)
    items = [(ip, mp) for (ip, lab, mp) in ds.samples if lab != normal_idx and mp and os.path.exists(mp)]
    S = UNET_IMG

    class SegDS(torch.utils.data.Dataset):
        def __init__(self, it): self.it = it; self.ti = T.Compose([T.Resize((S, S)), T.ToTensor()])
        def __len__(self): return len(self.it)
        def __getitem__(self, i):
            ip, mp = self.it[i]
            im = self.ti(Image.open(ip).convert("RGB"))
            mk = (self.ti(Image.open(mp).convert("L")) > 0.5).float()
            return im, mk

    class UNet(nn.Module):
        def __init__(self, c=16):
            super().__init__()
            def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                                                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())
            self.e1, self.e2, self.e3 = blk(3, c), blk(c, c*2), blk(c*2, c*4)
            self.pool = nn.MaxPool2d(2); self.b = blk(c*4, c*8)
            self.u3 = nn.ConvTranspose2d(c*8, c*4, 2, 2); self.d3 = blk(c*8, c*4)
            self.u2 = nn.ConvTranspose2d(c*4, c*2, 2, 2); self.d2 = blk(c*4, c*2)
            self.u1 = nn.ConvTranspose2d(c*2, c, 2, 2);   self.d1 = blk(c*2, c)
            self.o = nn.Conv2d(c, 1, 1)
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            b = self.b(self.pool(e3))
            d = self.d3(torch.cat([self.u3(b), e3], 1)); d = self.d2(torch.cat([self.u2(d), e2], 1))
            d = self.d1(torch.cat([self.u1(d), e1], 1)); return self.o(d)

    def dloss(lg, t):
        p = torch.sigmoid(lg).view(lg.size(0), -1); t = t.view(t.size(0), -1)
        return (1 - (2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()

    @torch.no_grad()
    def ev(net, ld):
        net.eval(); s = n = 0
        for im, mk in ld:
            im, mk = im.to(device), mk.to(device)
            p = (torch.sigmoid(net(im)) > 0.5).float().view(im.size(0), -1); t = mk.view(mk.size(0), -1)
            s += ((2*(p*t).sum(1)+1e-6)/(p.sum(1)+t.sum(1)+1e-6)).sum().item(); n += im.size(0)
        return s/max(n, 1)

    idx = np.arange(len(items)); rng = np.random.default_rng(42); rng.shuffle(idx)
    nt = max(2, int(len(idx)*UNET_TESTFRAC)); test, train = idx[:nt], idx[nt:]
    tl = torch.utils.data.DataLoader(SegDS([items[i] for i in test]), batch_size=UNET_BS)
    rows = []
    for frac in UNET_FRACTIONS:
        torch.manual_seed(0)
        k = max(2, int(len(train)*frac)); sub = rng.choice(train, size=min(k, len(train)), replace=False)
        trl = torch.utils.data.DataLoader(SegDS([items[i] for i in sub]), batch_size=UNET_BS, shuffle=True)
        net = UNet().to(device); opt = torch.optim.Adam(net.parameters(), 1e-3); bce = nn.BCEWithLogitsLoss()
        for _ in range(UNET_EPOCHS):
            net.train()
            for im, mk in trl:
                im, mk = im.to(device), mk.to(device)
                lg = net(im); loss = bce(lg, mk) + dloss(lg, mk)
                opt.zero_grad(); loss.backward(); opt.step()
        d = ev(net, tl); rows.append([f"{int(frac*100)}%", len(sub), round(d, 4)])
        log(f"   U-Net @ {int(frac*100)}% ({len(sub)} imgs): Dice {d:.4f}")
    write_csv("table4_baselines", rows, ["supervision", "n_train", "unet_dice"]); mark_done("table4_baselines")

# =============================================================== EXP2 Variant D
def _entropy(a):
    a = np.abs(np.asarray(a, float)); s = a.sum()
    if s <= 0: return 1.0
    p = a.ravel()/s; p = p[p > 0]; return float(-(p*np.log(p)).sum()/np.log(len(p)))

def exp_variantD():
    import torch, torch.nn as nn
    from PIL import Image
    from captum.attr import Saliency
    P = pipe(); L, sam, device, ds = P["L"], P["sam"], P["device"], P["ds"]
    diff = P["model"]
    idxs = lesion_indices(ds, N_VARIANTD, seed=1)
    try:
        import torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights
    except Exception as e:
        log(f"   [skip variantD] {e}"); return

    class Std(nn.Module):
        def __init__(self, dev):
            super().__init__(); self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
            self.preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]); self.num_classes = 1000
        def forward(self, x): return self.net(x)
        def predict_with_uncertainty(self, x):
            with torch.no_grad():
                pr = torch.softmax(self.net(x), 1); e = -(pr*torch.log(pr+1e-9)).sum(1)
            return pr.argmax(1), e, pr
    std = Std(device)

    rows = []
    for model, tag, force in [(std, "VariantD_standard", False), (diff, "Differential_indomain", True)]:
        sf = L.REVISION_CONFIG.get("KVASIR_FORCE_POLYP_CLASS", True)
        L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = force
        dices, zeros, ents = [], [], []
        try:
            for i in idxs:
                try:
                    _, mask_t, path, label = ds[i]
                    if force: L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(label)
                    # PER-MODEL preprocessing: build the tensor with THIS model's own transform so
                    # the standard ResNet is not fed BiomedCLIP-normalised (OOD) pixels. The original
                    # mismatch flattened the ResNet gradients -> high-entropy map -> whole-image bbox
                    # -> SAM box-leakage that artificially inflated the standard model's Dice.
                    img_m = model.preprocess(Image.open(path).convert("RGB"))
                    x = img_m.unsqueeze(0).to(device).requires_grad_(True)
                    with torch.no_grad(): tgt = int(model(x).argmax(1).item())
                    ents.append(_entropy(Saliency(model).attribute(x, target=tgt, abs=True).abs().sum(1).squeeze().detach().cpu().numpy()))
                    d, _ = L._kvasir_segment_metrics_only((img_m, mask_t, path), model, sam, device,
                                                          XAI_WEIGHTS, fusion_strategy="weighted_average", use_iterative=True)
                    dices.append(float(d)); zeros.append(int(d < 1e-6))
                except Exception as e:
                    log(f"   [warn] variantD {tag} {i}: {e}")
        finally:
            L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = sf
        rows.append([tag, round(np.mean(dices), 4) if dices else 0.0,
                     round(np.mean(zeros), 3) if zeros else 1.0,
                     round(np.mean(ents), 3) if ents else 1.0, len(dices)])
        log(f"   {tag}: Dice {rows[-1][1]} | zero_rate {rows[-1][2]} | entropy {rows[-1][3]}")
    write_csv("variantD_evidence", rows, ["backbone", "mean_dice", "zero_dice_rate", "attr_entropy_norm", "n"]); mark_done("variantD")

# =============================================================== EXP3 sensitivity
def exp_sensitivity():
    P = pipe(); ds = P["ds"]
    idxs = lesion_indices(ds, N_LENAS, seed=7); rows = []
    def rec(g, s, out):
        d = np.array([o[0] for o in out])
        rows.append([g, s, round(d.mean(), 4) if d.size else 0.0, round(d.std(), 4) if d.size else 0.0, d.size])
        log(f"   [{g}] {s}: Dice {rows[-1][2]} +/- {rows[-1][3]}")
    for strat in ["weighted_average", "multiplicative_consensus", "weighted_geometric_mean", "rank_aggregation"]:
        rec("fusion_strategy", strat, run_busi_subset(idxs, fusion=strat))
    for name, w in {"default": XAI_WEIGHTS, "equal": {"Saliency": 1/3, "IntegratedGradients": 1/3, "GradientShap": 1/3},
                    "saliency_heavy": {"Saliency": 0.6, "IntegratedGradients": 0.2, "GradientShap": 0.2},
                    "IG_heavy": {"Saliency": 0.2, "IntegratedGradients": 0.6, "GradientShap": 0.2}}.items():
        rec("xai_weights_Eq13", name, run_busi_subset(idxs, xai_weights=w))
    for pct in [50, 60, 70]:
        rec("sharpen_pct", f"pct={pct}", run_busi_subset(idxs, overrides={"XAI_SHARPEN_PERCENTILE": pct}))
    for tmax in [1, 3, 5]:
        rec("Tmax", f"Tmax={tmax}", run_busi_subset(idxs, overrides={"TMAX": tmax}))
    write_csv("sensitivity", rows, ["group", "setting", "mean_dice", "std_dice", "n"]); mark_done("sensitivity")

# =============================================================== EXP4 recent methods
def exp_recent_methods():
    P = pipe(); L, sam, ds, device = P["L"], P["sam"], P["ds"], P["device"]
    from PIL import Image
    idxs = lesion_indices(ds, min(N_LENAS, 120), seed=3); rows = []
    out = run_busi_subset(idxs)
    rows.append(["LENAS (ours, this subset)", round(np.mean([o[0] for o in out]), 4) if out else 0.0, "annotation-free", "this work"])
    try:
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        pr = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        cs = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
        dd = []
        for i in idxs:
            ip, lab, mp = ds.samples[i]
            img = Image.open(ip).convert("RGB")
            gt = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8) if mp and os.path.exists(mp) else None
            if gt is None: continue
            with torch.no_grad():
                inp = pr(text=["tumor"], images=[img], return_tensors="pt").to(device)
                p = torch.sigmoid(cs(**inp).logits)[0].cpu().numpy()
            import cv2
            pm = (cv2.resize(p, (gt.shape[1], gt.shape[0])) > 0.5).astype(np.uint8)
            dd.append(float(L.calculate_dice_score(pm, gt)))
        rows.append(["CLIPSeg zero-shot ('tumor')", round(np.mean(dd), 4) if dd else 0.0, "zero-shot", "run here"])
    except Exception as e:
        log(f"   [skip CLIPSeg] {e}")
    for name, v in REPORTED_NUMBERS.items():
        rows.append([name, v.get("busi_dice"), v.get("annotation", "?"), v.get("source", "reported")])
    write_csv("recent_methods", rows, ["method", "busi_dice", "annotation_type", "source"]); mark_done("recent_methods")

# =============================================================== EXP5 variability
def exp_variability():
    P = pipe(); L, ds = P["L"], P["ds"]
    idxs = lesion_indices(ds, N_LENAS, seed=11); rows = []; per = []
    for s in SEEDS:
        (L.set_global_seed if hasattr(L, "set_global_seed") else (lambda z: None))(s)
        out = run_busi_subset(idxs); d = np.array([o[0] for o in out])
        per.append(float(d.mean()) if d.size else 0.0)
        rows.append([f"seed_{s}", round(per[-1], 4), d.size]); log(f"   seed {s}: Dice {per[-1]:.4f}")
    if per:
        a = np.array(per)
        rows.append(["MEAN_ACROSS_RUNS", round(a.mean(), 4), len(per)])
        rows.append(["STD_ACROSS_RUNS", round(a.std(ddof=1) if len(a) > 1 else 0.0, 4), len(per)])
        log(f"   >> {a.mean():.4f} +/- {a.std(ddof=1) if len(a)>1 else 0:.4f}")
    write_csv("variability", rows, ["run", "mean_dice", "n"]); mark_done("variability")

# =============================================================== EXP6 CRF
def exp_crf_ablation():
    P = pipe(); ds = P["ds"]
    idxs = lesion_indices(ds, N_LENAS, seed=5); rows = []
    for crf in [False, True]:
        out = run_busi_subset(idxs, overrides={"USE_CRF": crf})
        allm, per = by_class(out, ds)
        rows.append(["with_CRF" if crf else "without_CRF", round(allm, 4),
                     round(per.get("benign", (0,))[0], 4), round(per.get("malignant", (0,))[0], 4), len(out)])
        log(f"   CRF={'on' if crf else 'off'}: all {rows[-1][1]} | benign {rows[-1][2]} | malignant {rows[-1][3]}")
    write_csv("crf_ablation", rows, ["setting", "mean_dice", "benign_dice", "malignant_dice", "n"]); mark_done("crf_ablation")

def export_excel():
    try: import pandas as pd
    except Exception: return
    x = os.path.join(OUT_DIR, "source_data.xlsx")
    try:
        with pd.ExcelWriter(x) as xw:
            for n in ["table4_baselines", "variantD_evidence", "sensitivity", "recent_methods", "variability", "crf_ablation", "component_ablation", "paper_variants"]:
                if os.path.exists(rp(n)): pd.read_csv(rp(n)).to_excel(xw, sheet_name=n[:31], index=False)
        log(f"[excel] -> {x}")
    except Exception as e:
        log(f"[excel] {e}")

def exp_component_ablation():
    """Leave-one-out ablation from the BUSI reference config (plain pipeline). Toggling a
    component shows its effect on BUSI -- e.g. enabling growth/candidate-selection HURTS BUSI,
    which explains why BUSI needs the plain config while Kvasir needs the aggressive one."""
    P = pipe(); L, ds = P["L"], P["ds"]
    idxs = lesion_indices(ds, N_LENAS, seed=13)
    comps = [("SmoothGrad", "XAI_SMOOTHGRAD", True),
             ("Fused-map sharpening", "XAI_SHARPEN", True),
             ("Reward candidate selection", "KVASIR_CANDIDATE_SELECTION", True),
             ("Blob coverage", "KVASIR_BLOB_COVERAGE", True),
             ("Recall growth", "KVASIR_RECALL_GROWTH", True),
             ("Snake non-erosion guard", "KVASIR_SNAKE_NONEROSION", True),
             ("Iterative init-mask fix", "KVASIR_ITER_INIT_MASK", True),
             ("CRF post-processing", "USE_CRF", False)]
    def md(ov):
        out = run_busi_subset(idxs, overrides=ov)
        return round(float(np.mean([o[0] for o in out])), 4) if out else 0.0
    base = md({}); rows = [["Full model (BUSI plain config)", "", base, 0.0, len(idxs)]]
    log(f"   Full (plain): {base}")
    for name, flag, default in comps:
        cur = L.REVISION_CONFIG.get(flag, default); d = md({flag: (not cur)})
        rows.append([f"toggle {name}", f"{flag}={not cur}", d, round(d - base, 4), len(idxs)])
        log(f"   toggle {name}: {d} ({d-base:+.4f})")
    write_csv("component_ablation", rows, ["config", "toggle", "mean_dice", "delta_vs_full", "n"]); mark_done("component_ablation")

def exp_paper_variants():
    """Paper's original ablation variants A/B/C (+ Full) on BUSI (plain reference config)."""
    P = pipe(); ds = P["ds"]
    idxs = lesion_indices(ds, N_LENAS, seed=17)
    def md(use_iter=True, ov=None):
        out = run_busi_subset(idxs, use_iter=use_iter, overrides=ov)
        d = np.array([o[0] for o in out]); i = np.array([o[1] for o in out])
        return (round(float(d.mean()), 4) if d.size else 0.0, round(float(i.mean()), 4) if i.size else 0.0)
    rows = []
    for tag, ui, ov in [("Full model", True, {}),
                        ("Variant A: no iterative self-correction", False, {}),
                        ("Variant B: no refinement (no snake + no iterative)", False, {"KVASIR_USE_SNAKE": False}),
                        ("Variant C: no SAM (saliency thresholding)", True, {"KVASIR_USE_SAM": False})]:
        dd, ii = md(ui, ov); rows.append([tag, dd, ii, len(idxs)])
        log(f"   {tag}: Dice {dd} | IoU {ii}")
    write_csv("paper_variants", rows, ["variant", "mean_dice", "mean_iou", "n"]); mark_done("paper_variants")

EXPS = {"table4_baselines": exp_table4_baselines, "variantD": exp_variantD, "sensitivity": exp_sensitivity,
        "recent_methods": exp_recent_methods, "variability": exp_variability, "crf_ablation": exp_crf_ablation, "component_ablation": exp_component_ablation, "paper_variants": exp_paper_variants}

def main():
    global N_LENAS, N_VARIANTD, SEEDS, UNET_FRACTIONS, UNET_EPOCHS, UNET_IMG
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        N_LENAS = 8; N_VARIANTD = 8; SEEDS = [0]; UNET_FRACTIONS = [0.01, 1.0]; UNET_EPOCHS = 2; UNET_IMG = 64
    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    log("=" * 70); log(f"LENAS BUSI experiments | smoke={a.smoke} | out={OUT_DIR}"); log("=" * 70)
    todo = [e.strip() for e in a.only.split(",") if e.strip()] or list(EXPS.keys())
    st = load_state()
    for n in todo:
        if n not in EXPS: log(f"[skip] unknown {n}"); continue
        done_csv = "variantD_evidence" if n == "variantD" else n
        if (not a.force) and n in st.get("done", []) and os.path.exists(rp(done_csv)):
            log(f"[resume] {n} done -> skip"); continue
        log(f"\n>>> RUNNING {n}"); t0 = time.time()
        try: EXPS[n]()
        except Exception as e:
            import traceback; log(f"[ERROR] {n}: {e}"); traceback.print_exc()
        st = load_state()
    export_excel()
    log(f"\nDone. Source data in {OUT_DIR}/source_data.xlsx (+ results/*.csv).")

if __name__ == "__main__":
    main()
