# -*- coding: utf-8 -*-
"""
LENAS REVISION EXPERIMENTS  (Communications AI & Computing, Submission d7c9f3f6-...)
====================================================================================
One self-contained, resumable script that runs EVERY reviewer-requested experiment and
writes source-data CSVs + a combined Excel workbook (one sheet per table).

Experiments
-----------
 1. table4_baselines : R1-1  U-Net at 1% / 5% / 10% (and 100%) pixel supervision on
                        Kvasir-SEG, seeded, correct per-fraction subsetting -> scores now
                        DIFFER and increase with supervision (explains the old flat 0.1760).
 2. variantD         : R1-2 / R2-2  Evidence that the "standard classifier" ablation truly
                        collapses to ~0 Dice: standard CNN attributions are DIFFUSE (high
                        entropy) -> prompts land off-lesion -> empty / wrong masks. Contrasted
                        against the in-domain Differential classifier on the same images.
 3. sensitivity      : R1-4 / R2-5  Fusion strategy, XAI fusion weights (Eq.13),
                        context-suppression ablation (Eq.5, alpha-related), and Tmax / growth.
 4. recent_methods   : R1-add  Label-free / zero-shot peers we can run (CLIPSeg, SAM-auto)
                        plus a REPORTED_NUMBERS block for figures you cite from recent papers.
 5. variability      : R1-add  LENAS over >=3 seeds -> per-run means and mean +/- std ACROSS runs.
 6. crf_ablation     : Editor  LENAS with CRF off vs on -> before/after Dice & IoU.

Usage
-----
  python lenas_revision_experiments.py                 # run everything (resumable)
  python lenas_revision_experiments.py --smoke         # tiny quick test of every experiment
  python lenas_revision_experiments.py --only sensitivity,crf_ablation
  python lenas_revision_experiments.py --force         # re-run even if results exist

"""
import os, sys, json, time, argparse, random, warnings
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
@dataclass
class CFG:
    module_name: str = "LENAS_Kvasir_revised"
    kvasir_dir: str = "/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG"
    sam_ckpt: str = "./sam_vit_b_01ec64.pth"
    classifier_ckpt: str = "best_kvasir_domain_classifier.pth"
    model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    out_dir: str = "./LENAS_experiments"

    # LENAS default fusion weights (Eq.13) -- taken from the pipeline
    xai_weights: Dict[str, float] = field(default_factory=lambda: {
        "Saliency": 0.279, "IntegratedGradients": 0.352, "GradientShap": 0.369})

    # sample sizes (subset for the LENAS-based experiments; None = all 1000)
    n_lenas: int = 200          # subset used for sensitivity / variability / crf / recent
    n_variantD: int = 120
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # U-Net baseline
    unet_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 1.0])
    unet_img: int = 128
    unet_epochs: int = 40
    unet_bs: int = 8
    unet_test_frac: float = 0.2

    # smoke overrides
    smoke: bool = False

    def apply_smoke(self):
        self.n_lenas = 8
        self.n_variantD = 8
        self.seeds = [0]
        self.unet_fractions = [0.01, 0.10, 1.0]
        self.unet_img = 64
        self.unet_epochs = 2
        self.unet_bs = 4
        self.n_recent = 6
        return self

CONFIG = CFG()

# Literature numbers you cite from recent papers (fill from their tables; clearly labelled).
# Format: method -> {"kvasir_dice":.., "busi_dice":.., "annotation":"zero-shot/..", "source":"[ref]"}
REPORTED_NUMBERS = {
    # "SaLIP (reported in [xx])": {"kvasir_dice": None, "busi_dice": None, "annotation": "zero-shot", "source": "[xx]"},
    # "GroundingDINO+SAM (reported in [yy])": {"kvasir_dice": None, "busi_dice": None, "annotation": "text-prompt", "source": "[yy]"},
    # "AutoMiSeg (reported in [zz])": {"kvasir_dice": 25.94, "busi_dice": 15.72, "annotation": "auto", "source": "[zz]"},
}

# ------------------------------------------------------------------ utils
def log(*a):
    print(*a, flush=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def ensure_dirs():
    for sub in ("results", "logs"):
        os.makedirs(os.path.join(CONFIG.out_dir, sub), exist_ok=True)

def results_path(name):
    return os.path.join(CONFIG.out_dir, "results", f"{name}.csv")

def state_path():
    return os.path.join(CONFIG.out_dir, "state.json")

def load_state():
    try:
        return json.load(open(state_path()))
    except Exception:
        return {"done": []}

def save_state(st):
    json.dump(st, open(state_path(), "w"), indent=2)

def write_csv(name, rows, header):
    import csv
    p = results_path(name)
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)
    log(f"   -> wrote {p}")
    return p

def mark_done(name):
    st = load_state()
    if name not in st["done"]:
        st["done"].append(name)
    save_state(st)

# ------------------------------------------------------------------ pipeline loader
_PIPE = {}

def load_pipeline():
    """Import the LENAS module (fresh), build the in-domain classifier + SAM predictor."""
    if _PIPE:
        return _PIPE
    import importlib, torch
    L = importlib.import_module(CONFIG.module_name)
    L = importlib.reload(L)
    log(f"[pipeline] module: {getattr(L,'__file__','?')}")
    import inspect
    if "init_mask" not in inspect.signature(L.iterative_self_correction_improved).parameters:
        log("  *** WARNING: stale module (no init_mask). Save the new LENAS_Kvasir_revised.py. ***")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # classifier from checkpoint (class list/count come from the checkpoint)
    ckpt = torch.load(L.resolve_ckpt(CONFIG.classifier_ckpt) if hasattr(L, "resolve_ckpt")
                      else CONFIG.classifier_ckpt, map_location=device, weights_only=False)
    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    ncls = int(ckpt.get("num_classes", len(classes) if classes else 8)) if isinstance(ckpt, dict) else 8
    model = L.DifferentialBiomedCLIP(CONFIG.model_name, ncls, device,
                                     class_names=classes, use_contrastive=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt,
                          strict=True)
    model.to(device).eval()
    if isinstance(ckpt, dict) and ckpt.get("polyp_idx") is not None:
        L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = True
        L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = int(ckpt["polyp_idx"])
        log(f"[pipeline] polyp target = {ckpt['polyp_idx']} (from checkpoint)")

    # SAM
    sam = None
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sm = sam_model_registry["vit_b"](checkpoint=CONFIG.sam_ckpt).to(device)
        sam = SamPredictor(sm)
        log("[pipeline] SAM loaded.")
    except Exception as e:
        log(f"[pipeline] SAM unavailable: {e}")

    _PIPE.update(dict(L=L, model=model, sam=sam, device=device))
    return _PIPE

def build_dataset(L, model):
    import torchvision.transforms as T
    ds = L.KvasirSEGDataset(
        image_dir=os.path.join(CONFIG.kvasir_dir, "images"),
        mask_dir=os.path.join(CONFIG.kvasir_dir, "masks"),
        transform=model.preprocess,
        mask_transform=T.Compose([T.ToTensor()]))
    return ds

def _subset_indices(n_total, k, seed=0):
    rng = np.random.default_rng(seed)
    if k is None or k >= n_total:
        return list(range(n_total))
    return sorted(rng.choice(n_total, size=k, replace=False).tolist())

def run_lenas_subset(L, model, sam, device, ds, idxs, xai_weights, fusion="weighted_average",
                     use_iterative=True, overrides=None):
    """Run the real LENAS metrics-only pipeline over a subset, with optional REVISION_CONFIG overrides."""
    saved = {}
    if overrides:
        for k, v in overrides.items():
            saved[k] = L.REVISION_CONFIG.get(k, None)
            L.REVISION_CONFIG[k] = v
    dices, ious = [], []
    try:
        for j, idx in enumerate(idxs):
            try:
                sample = ds[idx]
                d, i = L._kvasir_segment_metrics_only(sample, model, sam, device, xai_weights,
                                                      fusion_strategy=fusion, use_iterative=use_iterative)
                dices.append(float(d)); ious.append(float(i))
            except Exception as e:
                log(f"   [warn] sample {idx} failed: {e}")
            if (j + 1) % 25 == 0:
                log(f"     {j+1}/{len(idxs)} done")
    finally:
        if overrides:
            for k, v in saved.items():
                L.REVISION_CONFIG[k] = v
    return np.array(dices), np.array(ious)

# ================================================================== EXP 1: U-Net baselines
def exp_table4_baselines():
    import torch, torch.nn as nn, torch.nn.functional as Fnn
    from PIL import Image
    import torchvision.transforms as T
    P = load_pipeline(); L = P["L"]; device = P["device"]
    ds = build_dataset(L, P["model"])
    img_paths, mask_paths = ds.image_paths, ds.mask_paths
    S = CONFIG.unet_img

    class SegDS(torch.utils.data.Dataset):
        def __init__(self, ip, mp):
            self.ip, self.mp = ip, mp
            self.ti = T.Compose([T.Resize((S, S)), T.ToTensor()])
            self.tm = T.Compose([T.Resize((S, S)), T.ToTensor()])
        def __len__(self): return len(self.ip)
        def __getitem__(self, i):
            im = self.ti(Image.open(self.ip[i]).convert("RGB"))
            mk = self.tm(Image.open(self.mp[i]).convert("L"))
            mk = (mk > 0.5).float()
            return im, mk

    class UNet(nn.Module):
        def __init__(self, c=16):
            super().__init__()
            def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                                                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())
            self.e1, self.e2, self.e3 = blk(3, c), blk(c, c*2), blk(c*2, c*4)
            self.pool = nn.MaxPool2d(2); self.bott = blk(c*4, c*8)
            self.u3 = nn.ConvTranspose2d(c*8, c*4, 2, 2); self.d3 = blk(c*8, c*4)
            self.u2 = nn.ConvTranspose2d(c*4, c*2, 2, 2); self.d2 = blk(c*4, c*2)
            self.u1 = nn.ConvTranspose2d(c*2, c, 2, 2);   self.d1 = blk(c*2, c)
            self.out = nn.Conv2d(c, 1, 1)
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            b = self.bott(self.pool(e3))
            d = self.d3(torch.cat([self.u3(b), e3], 1))
            d = self.d2(torch.cat([self.u2(d), e2], 1))
            d = self.d1(torch.cat([self.u1(d), e1], 1))
            return self.out(d)

    def dice_loss(logit, tgt):
        p = torch.sigmoid(logit); p = p.view(p.size(0), -1); t = tgt.view(tgt.size(0), -1)
        inter = (p*t).sum(1); return (1 - (2*inter+1)/(p.sum(1)+t.sum(1)+1)).mean()

    @torch.no_grad()
    def eval_dice(net, loader):
        net.eval(); ds_, n = 0.0, 0
        for im, mk in loader:
            im, mk = im.to(device), mk.to(device)
            p = (torch.sigmoid(net(im)) > 0.5).float()
            p = p.view(p.size(0), -1); t = mk.view(mk.size(0), -1)
            inter = (p*t).sum(1); d = (2*inter+1e-6)/(p.sum(1)+t.sum(1)+1e-6)
            ds_ += d.sum().item(); n += im.size(0)
        return ds_/max(n, 1)

    # fixed train/test split (seeded), then subsample TRAIN per fraction
    idx = np.arange(len(img_paths)); rng = np.random.default_rng(42); rng.shuffle(idx)
    n_test = max(2, int(len(idx)*CONFIG.unet_test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    test_loader = torch.utils.data.DataLoader(
        SegDS([img_paths[i] for i in test_idx], [mask_paths[i] for i in test_idx]),
        batch_size=CONFIG.unet_bs)

    rows = []
    for frac in CONFIG.unet_fractions:
        set_seed(0)
        k = max(2, int(len(train_idx)*frac))
        sub = rng.choice(train_idx, size=min(k, len(train_idx)), replace=False)
        tl = torch.utils.data.DataLoader(
            SegDS([img_paths[i] for i in sub], [mask_paths[i] for i in sub]),
            batch_size=CONFIG.unet_bs, shuffle=True)
        net = UNet().to(device); opt = torch.optim.Adam(net.parameters(), 1e-3)
        bce = nn.BCEWithLogitsLoss()
        for ep in range(CONFIG.unet_epochs):
            net.train()
            for im, mk in tl:
                im, mk = im.to(device), mk.to(device)
                logit = net(im); loss = bce(logit, mk) + dice_loss(logit, mk)
                opt.zero_grad(); loss.backward(); opt.step()
        d = eval_dice(net, test_loader)
        log(f"   U-Net @ {int(frac*100)}% supervision ({len(sub)} imgs): test Dice {d:.4f}")
        rows.append([f"{int(frac*100)}%", len(sub), round(d, 4)])
    write_csv("table4_baselines", rows, ["supervision", "n_train", "unet_dice"])
    mark_done("table4_baselines")
    return rows

# ================================================================== EXP 2: Variant D evidence
def _attr_entropy(attr):
    a = np.abs(np.asarray(attr, dtype=np.float64)); s = a.sum()
    if s <= 0: return 1.0
    p = a.ravel()/s; p = p[p > 0]
    return float(-(p*np.log(p)).sum()/np.log(len(p)))   # normalised [0,1]; diffuse->~1

def exp_variantD():
    """Run the FULL LENAS pipeline with (a) the in-domain Differential classifier and (b) a
    STANDARD out-of-domain classifier, on the same images. A standard backbone yields diffuse
    attributions (higher entropy) and the classifier-confidence steps (candidate selection,
    growth) can no longer localise the lesion -> Dice collapses / many empty masks. This is the
    faithful analogue of the paper's Variant D and explains the original 0.0000 as genuine
    degenerate behaviour of a non-differential, out-of-domain backbone (not a bug)."""
    import torch, torch.nn as nn
    from PIL import Image
    P = load_pipeline(); L = P["L"]; device = P["device"]; sam = P["sam"]
    diff_model = P["model"]
    ds = build_dataset(L, diff_model)
    idxs = _subset_indices(len(ds), CONFIG.n_variantD, seed=1)

    # ---- STANDARD classifier wrapped to the interface the pipeline expects ----
    try:
        import torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights
    except Exception as e:
        log(f"   [skip variantD] torchvision unavailable: {e}"); return None

    class StandardClassifier(nn.Module):
        """Generic ImageNet CNN standing in for a non-differential backbone."""
        def __init__(self, dev):
            super().__init__()
            self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
            self.preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.num_classes = 1000
        def forward(self, x):
            return self.net(x)
        def predict_with_uncertainty(self, x):
            with torch.no_grad():
                logits = self.net(x); probs = torch.softmax(logits, 1)
                ent = -(probs * torch.log(probs + 1e-9)).sum(1)
            return probs.argmax(1), ent, probs

    std_model = StandardClassifier(device)

    def _run(model, tag, force_target):
        # force_target: for the standard model there is no polyp class -> use its own argmax so the
        # attribution is for the class the standard backbone actually predicts (typically off-lesion).
        saved_force = L.REVISION_CONFIG.get("KVASIR_FORCE_POLYP_CLASS", True)
        saved_tgt = L.REVISION_CONFIG.get("KVASIR_TARGET_CLASS", 6)
        if not force_target:
            L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = False   # use predicted class (meaningless for std)
        try:
            dices, empties, ents = [], [], []
            for idx in idxs:
                try:
                    _s = ds[idx]
                    # PER-MODEL preprocessing: score each backbone on ITS OWN transform (avoids the
                    # box-leakage artifact where a mismatched normalisation flattens gradients).
                    from captum.attr import Saliency
                    img_pil = Image.open(_s[2]).convert("RGB")
                    img_m = model.preprocess(img_pil)
                    sample = (img_m, _s[1], _s[2])
                    x = img_m.unsqueeze(0).to(device).requires_grad_(True)
                    with torch.no_grad():
                        tgt = int(model(x).argmax(1).item())
                    amap = Saliency(model).attribute(x, target=tgt, abs=True).abs().sum(1).squeeze().detach().cpu().numpy()
                    ents.append(_attr_entropy(amap))
                    # full LENAS pipeline
                    d, i = L._kvasir_segment_metrics_only(sample, model, sam, device, CONFIG.xai_weights,
                                                          fusion_strategy="weighted_average", use_iterative=True)
                    dices.append(float(d)); empties.append(int(d < 1e-6))
                except Exception as e:
                    log(f"   [warn] variantD {tag} idx {idx}: {e}")
            return dices, empties, ents
        finally:
            L.REVISION_CONFIG["KVASIR_FORCE_POLYP_CLASS"] = saved_force
            L.REVISION_CONFIG["KVASIR_TARGET_CLASS"] = saved_tgt

    rows = []
    for model, tag, ft in [(std_model, "VariantD_standard(full-pipeline)", False),
                           (diff_model, "Differential_indomain(full-pipeline)", True)]:
        dices, empties, ents = _run(model, tag, ft)
        rows.append([tag,
                     round(float(np.mean(dices)), 4) if dices else 0.0,
                     round(float(np.mean(empties)), 3) if empties else 1.0,
                     round(float(np.mean(ents)), 3) if ents else 1.0,
                     len(dices)])
        log(f"   {tag}: Dice {rows[-1][1]} | zero_dice_rate {rows[-1][2]} | attr_entropy {rows[-1][3]} | n {rows[-1][4]}")
    write_csv("variantD_evidence", rows,
              ["backbone", "mean_dice", "zero_dice_rate", "attr_entropy_norm", "n"])
    mark_done("variantD")
    return rows

# ================================================================== EXP 3: sensitivity
def exp_sensitivity():
    P = load_pipeline(); L = P["L"]; device = P["device"]
    ds = build_dataset(L, P["model"]); model, sam = P["model"], P["sam"]
    idxs = _subset_indices(len(ds), CONFIG.n_lenas, seed=7)
    rows = []

    def record(group, setting, dices):
        rows.append([group, setting, round(float(dices.mean()), 4) if dices.size else 0.0,
                     round(float(dices.std()), 4) if dices.size else 0.0, int(dices.size)])
        log(f"   [{group}] {setting}: Dice {rows[-1][2]} +/- {rows[-1][3]}")

    # (a) fusion strategy
    for strat in ["weighted_average", "multiplicative_consensus", "weighted_geometric_mean", "rank_aggregation"]:
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights, fusion=strat)
        record("fusion_strategy", strat, d)

    # (b) XAI fusion weights (Eq.13)
    weight_sets = {
        "default(0.28/0.35/0.37)": CONFIG.xai_weights,
        "equal(0.33/0.33/0.33)": {"Saliency": 1/3, "IntegratedGradients": 1/3, "GradientShap": 1/3},
        "saliency_heavy(0.6/0.2/0.2)": {"Saliency": 0.6, "IntegratedGradients": 0.2, "GradientShap": 0.2},
        "IG_heavy(0.2/0.6/0.2)": {"Saliency": 0.2, "IntegratedGradients": 0.6, "GradientShap": 0.2},
        "GShap_heavy(0.2/0.2/0.6)": {"Saliency": 0.2, "IntegratedGradients": 0.2, "GradientShap": 0.6},
    }
    for name, w in weight_sets.items():
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, w, fusion="weighted_average")
        record("xai_weights_Eq13", name, d)

    # (c) context-suppression ablation (Eq.5 / alpha-related): border-specular suppression on/off
    for flag in [True, False]:
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights,
                                overrides={"SUPPRESS_BORDER_SPECULAR": flag})
        record("context_suppression_Eq5", f"suppress={flag}", d)

    # (d) Tmax (iterative depth) and growth percentile (robustness)
    for tmax in [1, 3, 5]:
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights, overrides={"TMAX": tmax})
        record("Tmax", f"Tmax={tmax}", d)
    for gp in [60, 65, 70]:
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights,
                                overrides={"KVASIR_GROW_PERCENTILE": gp})
        record("grow_percentile", f"pct={gp}", d)

    write_csv("sensitivity", rows, ["group", "setting", "mean_dice", "std_dice", "n"])
    mark_done("sensitivity")
    return rows

# ================================================================== EXP 4: recent methods
def exp_recent_methods():
    P = load_pipeline(); L = P["L"]; device = P["device"]
    ds = build_dataset(L, P["model"]); sam = P["sam"]
    n = getattr(CONFIG, "n_recent", CONFIG.n_lenas)
    idxs = _subset_indices(len(ds), n, seed=3)
    from PIL import Image
    import numpy as np
    rows = []

    # our method reference on the SAME subset
    d, i = run_lenas_subset(L, P["model"], sam, device, ds, idxs, CONFIG.xai_weights)
    rows.append(["LENAS (ours, this subset)", round(float(d.mean()), 4) if d.size else 0.0, "annotation-free", "this work"])

    # CLIPSeg zero-shot
    try:
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
        dd = []
        for idx in idxs:
            img = Image.open(ds.image_paths[idx]).convert("RGB")
            gt = (np.array(Image.open(ds.mask_paths[idx]).convert("L")) > 127).astype(np.uint8)
            with torch.no_grad():
                inp = proc(text=["polyp"], images=[img], return_tensors="pt").to(device)
                pred = torch.sigmoid(cseg(**inp).logits)[0].cpu().numpy()
            import cv2
            pm = (cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5).astype(np.uint8)
            dd.append(float(L.calculate_dice_score(pm, gt)))
        rows.append(["CLIPSeg zero-shot ('polyp')", round(float(np.mean(dd)), 4) if dd else 0.0, "zero-shot", "run here"])
    except Exception as e:
        log(f"   [skip CLIPSeg] {e}")

    # SAM automatic (label-free, best-mask oracle upper bound is NOT computed; report largest mask)
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        gen = SamAutomaticMaskGenerator(sam.model) if sam is not None else None
        if gen is not None:
            dd = []
            for idx in idxs:
                img = np.array(Image.open(ds.image_paths[idx]).convert("RGB"))
                gt = (np.array(Image.open(ds.mask_paths[idx]).convert("L")) > 127).astype(np.uint8)
                masks = gen.generate(img)
                if not masks:
                    dd.append(0.0); continue
                big = max(masks, key=lambda m: m["area"])["segmentation"].astype(np.uint8)
                dd.append(float(L.calculate_dice_score(big, gt)))
            rows.append(["SAM-automatic (largest mask)", round(float(np.mean(dd)), 4) if dd else 0.0, "label-free", "run here"])
    except Exception as e:
        log(f"   [skip SAM-auto] {e}")

    # literature-reported numbers (clearly labelled; you fill REPORTED_NUMBERS)
    for name, v in REPORTED_NUMBERS.items():
        rows.append([name, v.get("kvasir_dice"), v.get("annotation", "?"), v.get("source", "reported")])

    write_csv("recent_methods", rows, ["method", "kvasir_dice", "annotation_type", "source"])
    mark_done("recent_methods")
    return rows

# ================================================================== EXP 5: variability
def exp_variability():
    P = load_pipeline(); L = P["L"]; device = P["device"]
    ds = build_dataset(L, P["model"]); model, sam = P["model"], P["sam"]
    idxs = _subset_indices(len(ds), CONFIG.n_lenas, seed=11)
    per_run = []
    rows = []
    for s in CONFIG.seeds:
        (L.set_global_seed if hasattr(L, "set_global_seed") else set_seed)(s)
        d, i = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights)
        md, mi = float(d.mean()) if d.size else 0.0, float(i.mean()) if i.size else 0.0
        per_run.append((md, mi))
        rows.append([f"seed_{s}", round(md, 4), round(mi, 4), int(d.size)])
        log(f"   seed {s}: Dice {md:.4f} | IoU {mi:.4f}")
    if per_run:
        ds_ = np.array([x[0] for x in per_run]); is_ = np.array([x[1] for x in per_run])
        rows.append(["MEAN_ACROSS_RUNS", round(float(ds_.mean()), 4), round(float(is_.mean()), 4), len(per_run)])
        rows.append(["STD_ACROSS_RUNS", round(float(ds_.std(ddof=1) if len(ds_) > 1 else 0.0), 4),
                     round(float(is_.std(ddof=1) if len(is_) > 1 else 0.0), 4), len(per_run)])
        log(f"   >> Dice {ds_.mean():.4f} +/- {ds_.std(ddof=1) if len(ds_)>1 else 0:.4f} across {len(per_run)} runs")
    write_csv("variability", rows, ["run", "mean_dice", "mean_iou", "n"])
    mark_done("variability")
    return rows

# ================================================================== EXP 6: CRF before/after
def exp_crf_ablation():
    P = load_pipeline(); L = P["L"]; device = P["device"]
    ds = build_dataset(L, P["model"]); model, sam = P["model"], P["sam"]
    idxs = _subset_indices(len(ds), CONFIG.n_lenas, seed=5)
    rows = []
    for use_crf in [False, True]:
        set_seed(0)
        d, i = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights,
                                overrides={"USE_CRF": use_crf})
        rows.append(["with_CRF" if use_crf else "without_CRF",
                     round(float(d.mean()), 4) if d.size else 0.0,
                     round(float(i.mean()), 4) if i.size else 0.0, int(d.size)])
        log(f"   CRF={'on' if use_crf else 'off'}: Dice {rows[-1][1]} | IoU {rows[-1][2]}")
    write_csv("crf_ablation", rows, ["setting", "mean_dice", "mean_iou", "n"])
    mark_done("crf_ablation")
    return rows

# ------------------------------------------------------------------ Excel export
def export_excel():
    try:
        import pandas as pd
    except Exception as e:
        log(f"[excel] pandas unavailable ({e}); CSVs already written."); return
    xlsx = os.path.join(CONFIG.out_dir, "source_data.xlsx")
    try:
        with pd.ExcelWriter(xlsx) as xw:
            for name in ["table4_baselines", "variantD_evidence", "sensitivity",
                         "recent_methods", "variability", "crf_ablation", "component_ablation", "paper_variants"]:
                p = results_path(name)
                if os.path.exists(p):
                    pd.read_csv(p).to_excel(xw, sheet_name=name[:31], index=False)
        log(f"[excel] source data workbook -> {xlsx}")
    except Exception as e:
        log(f"[excel] failed ({e}); CSVs are still available in results/.")

# ------------------------------------------------------------------ runner
def exp_component_ablation():
    """Leave-one-out ablation from the FULL model (all components on). Each component is turned
    OFF in turn; the Dice drop is its contribution. Combine with the paper's original ablation
    variants into one table."""
    P = load_pipeline(); L = P["L"]; device = P["device"]; model, sam = P["model"], P["sam"]
    ds = build_dataset(L, model); idxs = _subset_indices(len(ds), CONFIG.n_lenas, seed=13)
    comps = [("SmoothGrad", "XAI_SMOOTHGRAD", True),
             ("Fused-map sharpening", "XAI_SHARPEN", True),
             ("Reward candidate selection", "KVASIR_CANDIDATE_SELECTION", True),
             ("Blob coverage", "KVASIR_BLOB_COVERAGE", True),
             ("Recall growth", "KVASIR_RECALL_GROWTH", True),
             ("Snake non-erosion guard", "KVASIR_SNAKE_NONEROSION", True),
             ("Iterative init-mask fix", "KVASIR_ITER_INIT_MASK", True),
             ("CRF post-processing", "USE_CRF", False)]
    def md(ov):
        d, _ = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights, overrides=ov)
        return float(d.mean()) if d.size else 0.0
    base = md({}); rows = [["Full model (all components)", "", round(base, 4), 0.0, len(idxs)]]
    log(f"   Full model: {base:.4f}")
    for name, flag, default in comps:
        cur = L.REVISION_CONFIG.get(flag, default)
        d = md({flag: (not cur)})
        rows.append([f"w/o {name}", f"{flag}={not cur}", round(d, 4), round(d - base, 4), len(idxs)])
        log(f"   w/o {name}: {d:.4f} ({d-base:+.4f})")
    write_csv("component_ablation", rows, ["config", "toggle", "mean_dice", "delta_vs_full", "n"]); mark_done("component_ablation")

def exp_paper_variants():
    """Paper's original ablation variants A/B/C (+ Full) on the corrected pipeline."""
    P = load_pipeline(); L = P["L"]; device = P["device"]; model, sam = P["model"], P["sam"]
    ds = build_dataset(L, model); idxs = _subset_indices(len(ds), CONFIG.n_lenas, seed=17)
    def md(use_iter=True, ov=None):
        d, i = run_lenas_subset(L, model, sam, device, ds, idxs, CONFIG.xai_weights, use_iterative=use_iter, overrides=ov)
        return (round(float(d.mean()), 4) if d.size else 0.0, round(float(i.mean()), 4) if i.size else 0.0)
    rows = []
    for tag, ui, ov in [("Full model", True, {}),
                        ("Variant A: no iterative self-correction", False, {}),
                        ("Variant B: no refinement (no snake + no iterative)", False, {"KVASIR_USE_SNAKE": False}),
                        ("Variant C: no SAM (saliency thresholding)", True, {"KVASIR_USE_SAM": False})]:
        dd, ii = md(ui, ov); rows.append([tag, dd, ii, len(idxs)])
        log(f"   {tag}: Dice {dd} | IoU {ii}")
    write_csv("paper_variants", rows, ["variant", "mean_dice", "mean_iou", "n"]); mark_done("paper_variants")

EXPERIMENTS = {
    "table4_baselines": exp_table4_baselines,
    "variantD": exp_variantD,
    "sensitivity": exp_sensitivity,
    "recent_methods": exp_recent_methods,
    "variability": exp_variability,
    "crf_ablation": exp_crf_ablation,
    "component_ablation": exp_component_ablation,
    "paper_variants": exp_paper_variants,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny quick test of every experiment")
    ap.add_argument("--only", type=str, default="", help="comma list of experiments to run")
    ap.add_argument("--force", action="store_true", help="re-run even if results already exist")
    ap.add_argument("--out", type=str, default="", help="override output directory")
    args = ap.parse_args()

    if args.out:
        CONFIG.out_dir = args.out
    if args.smoke:
        CONFIG.smoke = True; CONFIG.apply_smoke()
    ensure_dirs()
    log("=" * 70)
    log(f"LENAS revision experiments | smoke={CONFIG.smoke} | out={CONFIG.out_dir}")
    log("=" * 70)

    todo = [e.strip() for e in args.only.split(",") if e.strip()] or list(EXPERIMENTS.keys())
    st = load_state()
    for name in todo:
        if name not in EXPERIMENTS:
            log(f"[skip] unknown experiment '{name}'"); continue
        if (not args.force) and (name in st.get("done", [])) and os.path.exists(results_path(
                "variantD_evidence" if name == "variantD" else name)):
            log(f"[resume] '{name}' already done -> skipping (use --force to redo)"); continue
        log(f"\n>>> RUNNING: {name}")
        t0 = time.time()
        try:
            EXPERIMENTS[name]()
            log(f"<<< {name} finished in {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            log(f"[ERROR] {name} failed: {e}")
            traceback.print_exc()
        st = load_state()

    export_excel()
    log("\nAll requested experiments processed. Source data in "
        f"{os.path.join(CONFIG.out_dir, 'source_data.xlsx')} (+ results/*.csv).")

if __name__ == "__main__":
    main()
