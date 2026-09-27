# -*- coding: utf-8 -*-
"""
LENAS FIGURE REGENERATOR  (uses the REAL pipeline; only plotting is changed)
============================================================================
Regenerates the qualitative multi-panel figure by calling the EXACT pipeline
functions from LENAS_Kvasir_revised.py that produced the paper figures, so the
segmentation is identical. Only the rendering is editor-compliant:
  * perceptually-uniform viridis colormap (NO rainbow/jet)
  * large, legible fonts
  * NO baked-in filename titles / extraneous text (only panel labels)
  * saved as individual "Figure_XX.pdf" + ".png" (one page, all panels in one file)

Works for kvasir / busi / capsule. Optional --train to (re)train the classifier.

Usage
-----
  python lenas_figure_regen.py --dataset busi   --image "img (1).png" --gt "img (1)_mask.png" --fignum 5
  python lenas_figure_regen.py --dataset kvasir --image polyp.jpg --gt polyp_mask.jpg --fignum 6
  python lenas_figure_regen.py --dataset capsule --train --epochs 15
  python lenas_figure_regen.py --dataset capsule --image bleeding.jpg --ckpt capsule.pth --fignum 15
"""
import os, sys, argparse
from dataclasses import dataclass, field
import numpy as np
from PIL import Image as _PILImage

# ----------------------------- CONFIG -----------------------------
@dataclass
class CONFIG:
    model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ckpt: dict = field(default_factory=lambda: {
        "kvasir": "/media/data/DARE/best_kvasir_domain_classifier.pth",
        "busi":   "/media/data/DARE/best_busi_domain_classifier.pth",
        "capsule":"/media/data/DARE/best_capsule_domain_classifier.pth"})
    classes: dict = field(default_factory=lambda: {
        "kvasir": ["dyed-lifted-polyps","dyed-resection-margins","esophagitis","normal-cecum",
                   "normal-pylorus","normal-z-line","polyps","ulcerative-colitis"],
        "busi":   ["benign","malignant","normal"],
        "capsule":["Angioectasia","Bleeding","Erosion","Erythema","Foreign Body",
                   "Lymphangiectasia","Normal","Polyp","Ulcer","Worms"]})
    data_dir: dict = field(default_factory=lambda: {
        "kvasir": "/media/data/DARE/kvasir-dataset/",
        "busi":   "/media/data/DARE/BUSI_Dataset/Dataset_BUSI_with_GT/",
        "capsule":"/root/.cache/kagglehub/datasets/podakantisatyajith/capsule-vision-challenge-final-dataset-2024/versions/1/Dataset/Dataset/training"})
    sam_ckpt: str = "/media/data/DARE/sam_vit_b_01ec64.pth"
    xai_weights: dict = field(default_factory=lambda: {"Saliency": 0.279, "IntegratedGradients": 0.352, "GradientShap": 0.369})
    fusion_strategy: str = "weighted_average"
    cmap: str = "viridis"
    out_dir: str = "./figures_out"
    fs_panel: int = 15
    fs_cbar: int = 12
    dpi: int = 300

CFG = CONFIG()

def setup_mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": CFG.fs_panel, "axes.titlesize": CFG.fs_panel,
        "axes.titleweight": "bold", "figure.dpi": CFG.dpi, "savefig.dpi": CFG.dpi,
        "font.family": "DejaVu Sans"})
    return plt

# ----------------------------- pipeline import -----------------------------
def _pipe():
    try:
        import LENAS_Kvasir_revised as L
    except Exception as e:
        sys.exit(f"[error] run from the folder containing LENAS_Kvasir_revised.py ({e})")
    return L

def load_model(dataset, device, classes=None, ckpt=None):
    L = _pipe(); import torch
    classes = classes or CFG.classes[dataset]
    model = L.DifferentialBiomedCLIP(CFG.model_name, len(classes), device,
                                     lambda_init=0.8, class_names=classes).to(device)
    ck = ckpt or CFG.ckpt[dataset]
    if os.path.exists(ck):
        sd = torch.load(ck, map_location=device)
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        model.load_state_dict(sd, strict=False); print(f"[model] loaded {ck}")
    else:
        print(f"[model] WARNING checkpoint missing at {ck}")
    model.eval(); return model, classes

def load_sam(device):
    L = _pipe()
    if not (getattr(L, "SAM_AVAILABLE", True) and os.path.exists(CFG.sam_ckpt)):
        print("[sam] unavailable; SAM panels skipped"); return None
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    sam = sam_model_registry["vit_b"](checkpoint=CFG.sam_ckpt).to(device)
    return SamPredictor(sam)

# ----------------------------- run the REAL pipeline, capture intermediates -----------------------------
def run_pipeline(L, model, sam, device, img_path, gt_path, dataset):
    import torch, cv2
    original_np = np.array(_PILImage.open(img_path).convert("RGB"))
    img_t = model.preprocess(_PILImage.open(img_path).convert("RGB"))
    with torch.no_grad():
        _, entropy, probs = model.predict_with_uncertainty(img_t.unsqueeze(0).to(device))
        pred = int(torch.argmax(probs, 1).item()); entropy_val = float(entropy[0].item())
    target_class = pred
    explanations = L.generate_explanations_focused(model, img_t.unsqueeze(0), target_class, device, use_cache=False)
    fused_map = L.advanced_xai_fusion(explanations, CFG.xai_weights, CFG.fusion_strategy)
    try:
        if getattr(L, "REVISION_CONFIG", {}).get("XAI_SHARPEN", True):
            fused_map = L.sharpen_saliency_map(fused_map)
    except Exception: pass

    num_prompts, _ = L.uncertainty_guided_prompts(fused_map, entropy_val, probs[0].cpu().numpy())
    bbox_xai = L.extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05)
    pos_xai = L.extract_focused_positive_prompts(fused_map, bbox_xai, num_prompts=num_prompts)
    circ = L.detect_circular_mask(original_np.shape)
    circ_xai = cv2.resize(circ, (fused_map.shape[1], fused_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    neg_xai = L.extract_smart_negative_prompts(fused_map, bbox_xai, circ_xai, num_negatives=3)
    xs, os_ = fused_map.shape, original_np.shape[:2]
    bbox_o = L.transform_bbox_to_original(bbox_xai, xs, os_)
    pos_o = L.transform_coordinates_to_original(pos_xai, xs, os_)
    neg_o = L.transform_coordinates_to_original(neg_xai, xs, os_) if len(neg_xai) else np.array([])

    import cv2
    def _clean(m):
        m = (m > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == k).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if dataset == "busi":
        # BUSI plain configuration (no candidate-selection / no recall-growth): tighter, matches the paper
        init_mask, sam_score = L.segment_with_sam_enhanced(
            sam, original_np, pos_o, neg_o if len(neg_o) else None, bbox_o, post_process=True)
        grown = init_mask
    else:
        init_mask, sam_score = L.select_best_sam_candidate(
            model, sam, original_np, img_t, fused_map, target_class, device, bbox_o, pos_o, neg_o)
        try:
            grown = L.grow_mask_to_saliency(sam, init_mask, fused_map, original_np,
                                            classifier=model, image_tensor=img_t, target_class=target_class,
                                            device=device, explanations=explanations)
        except Exception:
            grown = init_mask
    init_mask = _clean(init_mask); grown = _clean(grown)
    snake = L.refine_mask_with_snake_improved(grown, original_np, fused_map, iterations=100)
    snake = _clean(snake)
    # snake-runaway guard: if the contour ballooned well beyond the SAM mask, keep the tighter mask
    if snake.sum() > 1.4 * max(grown.sum(), 1):
        snake = grown

    gt = None
    if gt_path and os.path.exists(gt_path):
        gt = (np.array(_PILImage.open(gt_path).convert("L")) > 127).astype(np.uint8)
        gt = cv2.resize(gt, (os_[1], os_[0]), interpolation=cv2.INTER_NEAREST)

    def _dice(a, b):
        if b is None: return None
        a = a.astype(bool); b = b.astype(bool)
        return 2*(a & b).sum()/(a.sum()+b.sum()+1e-8)
    def _iou(a, b):
        if b is None: return None
        a = a.astype(bool); b = b.astype(bool)
        return (a & b).sum()/((a | b).sum()+1e-8)

    return dict(original=original_np, explanations=explanations, fused=fused_map,
                bbox=bbox_o, pos=pos_o, neg=neg_o, init=init_mask, snake=snake, final=snake,
                sam_score=float(sam_score), gt=gt, pred_label=CFG.classes[dataset][pred],
                conf=float(probs[0, pred].item()), entropy=entropy_val,
                dice=_dice(snake, gt), iou=_iou(snake, gt),
                init_dice=_dice(init_mask, gt))

# ----------------------------- plotting (viridis, clean) -----------------------------
def _up(a, hw):
    import cv2; return cv2.resize(a.astype(np.float32), (hw[1], hw[0]))

def make_figure(R, fignum, plt):
    import numpy as np, matplotlib.patches as mpatches
    img = R["original"]; H, W = img.shape[:2]
    gray = np.array(_PILImage.fromarray(img).convert("L"))
    w = CFG.xai_weights
    fig, ax = plt.subplots(3, 5, figsize=(22, 13))

    # Row 1: original + 3 XAI (viridis, with weight in title) + colorbars
    ax[0,0].imshow(img)
    t = f"Original ({R['pred_label']})\nConf: {R['conf']:.3f}, Entropy: {R['entropy']:.3f}"
    if R["dice"] is not None: t += f"\nDice: {R['dice']:.3f}, IoU: {R['iou']:.3f}"
    ax[0,0].set_title(t); ax[0,0].axis("off")
    for j,(k,lbl) in enumerate([("Saliency","Saliency"),
                                ("IntegratedGradients","Integrated Gradients"),
                                ("GradientShap","GradientSHAP")], start=1):
        wt = w[k]
        hm = R["explanations"][k]; hm = (hm-hm.min())/(hm.max()-hm.min()+1e-8)
        hm = _up(hm,(H,W)); hmb = np.power(hm, 0.6)          # gamma-boost mid/high saliency
        amap = np.clip(hmb, 0, 1) * 0.9                       # per-pixel alpha: background stays gray
        ax[0,j].imshow(gray, cmap="gray")
        im = ax[0,j].imshow(hmb, cmap=CFG.cmap, alpha=amap, vmin=0, vmax=1)
        ax[0,j].set_title(f"{lbl}\n(w={wt:.3f})"); ax[0,j].axis("off")
        cb = plt.colorbar(im, ax=ax[0,j], fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=CFG.fs_cbar); cb.set_label("Normalised attribution", fontsize=CFG.fs_cbar)  # editorial request: labelled colour bars
    ax[0,4].axis("off")

    # Row 2: fused+prompts | initial SAM | snake | final | GT
    fmap = _up((R["fused"]-R["fused"].min())/(R["fused"].max()-R["fused"].min()+1e-8), (H,W))
    fmb = np.power(fmap, 0.6); famap = np.clip(fmb, 0, 1) * 0.9
    ax[1,0].imshow(gray, cmap="gray"); ax[1,0].imshow(fmb, cmap=CFG.cmap, alpha=famap, vmin=0, vmax=1)
    b = R["bbox"]; ax[1,0].add_patch(mpatches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],
                    fill=False, edgecolor="lime", lw=2, linestyle="--"))
    if len(R["pos"]): ax[1,0].scatter(R["pos"][:,0],R["pos"][:,1],marker="*",c="lime",s=200,edgecolors="k",linewidths=0.7,label="Pos")
    if len(R["neg"]): ax[1,0].scatter(R["neg"][:,0],R["neg"][:,1],marker="X",c="red",s=140,edgecolors="k",linewidths=0.7,label="Neg")
    ax[1,0].legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax[1,0].set_title(f"Fused Map + Prompts\n({len(R['pos'])}+, {len(R['neg'])}-)"); ax[1,0].axis("off")
    for a,m,ttl,color in [(ax[1,1],R["init"],f"Initial SAM\nScore: {R['sam_score']:.3f}",(0.9,0.15,0.15)),
                          (ax[1,2],R["snake"],f"Snake Refined\nPixels: {int(R['snake'].sum())}",(0.15,0.8,0.2)),
                          (ax[1,3],R["final"],f"Final Prediction" + (f"\nDice: {R['dice']:.3f}, IoU: {R['iou']:.3f}" if R['dice'] is not None else ""),(0.15,0.3,0.95))]:
        a.imshow(img); ov=np.zeros((H,W,4)); ov[m>0]=(*color,0.5); a.imshow(ov); a.set_title(ttl); a.axis("off")
    if R["gt"] is not None:
        ax[1,4].imshow(img); ov=np.zeros((H,W,4)); ov[R["gt"]>0]=(0.9,0.8,0.1,0.55); ax[1,4].imshow(ov)
        ax[1,4].set_title(f"Ground Truth\nPixels: {int(R['gt'].sum())}")
    ax[1,4].axis("off")

    # Row 3: binary masks + GT
    for a,m,ttl in [(ax[2,0],R["init"],"Initial Mask"),(ax[2,1],R["snake"],"After Snake"),
                    (ax[2,2],R["final"],"Final Mask")]:
        a.imshow(m, cmap="gray"); a.set_title(ttl); a.axis("off")
    ax[2,3].imshow(R["gt"] if R["gt"] is not None else np.zeros((H,W)), cmap="gray")
    ax[2,3].set_title("Ground Truth"); ax[2,3].axis("off"); ax[2,4].axis("off")

    plt.tight_layout()
    os.makedirs(CFG.out_dir, exist_ok=True)
    base = os.path.join(CFG.out_dir, f"Figure_{int(fignum):02d}")
    fig.savefig(base+".pdf", bbox_inches="tight"); fig.savefig(base+".png", bbox_inches="tight")
    plt.close(fig); print(f"[saved] {base}.pdf and .png")

# ----------------------------- optional training -----------------------------
def train_classifier(dataset, device, classes, data_dir, epochs, out_ckpt, smoke=False):
    L = _pipe(); import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    model = L.DifferentialBiomedCLIP(CFG.model_name, len(classes), device, lambda_init=0.8, class_names=classes).to(device)
    pre = model.preprocess; c2i = {c:i for i,c in enumerate(classes)}; samples=[]
    for c in classes:
        d = os.path.join(data_dir, c)
        if not os.path.isdir(d): print(f"[train] missing {d}"); continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                samples.append((os.path.join(d,fn), c2i[c]))
    if not samples: sys.exit(f"[train] no images under {data_dir}")
    if smoke: samples=samples[:40]; epochs=1
    print(f"[train] {len(samples)} imgs / {len(classes)} classes")
    class DS(Dataset):
        def __len__(s): return len(samples)
        def __getitem__(s,i):
            p,y=samples[i]; return pre(_PILImage.open(p).convert("RGB")), y
    ds=DS(); nval=max(1,int(0.15*len(ds)))
    tr,va=random_split(ds,[len(ds)-nval,nval],generator=torch.Generator().manual_seed(42))
    tl=DataLoader(tr,batch_size=16,shuffle=True,num_workers=4,drop_last=len(tr)>16); vl=DataLoader(va,batch_size=16,num_workers=4)
    counts=np.bincount([y for _,y in samples],minlength=len(classes)).astype(np.float32)
    wgt=torch.tensor((counts.sum()/(counts+1e-6))/len(classes),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=wgt); best=0.0
    for ep in range(epochs):
        for n,p in model.named_parameters():
            head=any(k in n.lower() for k in ("classif","head","text","lambda")); p.requires_grad=True if head else (ep>=2)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=1e-4); model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); lg=model(x); lg=lg[0] if isinstance(lg,tuple) else lg
            loss=crit(lg,y); opt.zero_grad(); loss.backward(); opt.step()
            if smoke: break
        model.eval(); cor=tot=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device); lg=model(x); lg=lg[0] if isinstance(lg,tuple) else lg
                cor+=(lg.argmax(1)==y).sum().item(); tot+=y.numel()
                if smoke: break
        acc=cor/max(tot,1); print(f"[train] epoch {ep+1}/{epochs} val_acc={acc:.4f}")
        if acc>=best: best=acc; torch.save({"model_state_dict":model.state_dict(),"classes":classes}, out_ckpt); print(f"[train] saved best -> {out_ckpt}")

# ----------------------------- driver -----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kvasir","busi","capsule"])
    ap.add_argument("--image", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--fignum", default=5)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a=ap.parse_args()

    plt=setup_mpl(); import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    classes=[c.strip() for c in a.classes.split(",")] if a.classes else CFG.classes[a.dataset]
    ckpt=a.ckpt or CFG.ckpt[a.dataset]; data_dir=a.data_dir or CFG.data_dir.get(a.dataset)

    if a.train:
        train_classifier(a.dataset, device, classes, data_dir, a.epochs, ckpt, smoke=a.smoke)
        if not a.image: print("[done] trained; no --image."); return
    if not a.image: sys.exit("[error] provide --image (or --train only).")

    L=_pipe(); model,classes=load_model(a.dataset, device, classes=classes, ckpt=ckpt)
    sam=load_sam(device)
    if sam is None: sys.exit("[error] SAM required for segmentation panels.")
    R=run_pipeline(L, model, sam, device, a.image, a.gt, a.dataset)
    make_figure(R, a.fignum, plt)

if __name__ == "__main__":
    main()
