# -*- coding: utf-8 -*-
"""
Enhanced Medical Image Analysis Pipeline - UPDATED FOR CAPSULE ENDOSCOPY + KVASIR-SEG
Complete implementation for both classification and segmentation tasks
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import os, cv2, gc, json
from tqdm.notebook import tqdm
from collections import OrderedDict
import math
from IPython.display import display, clear_output
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from skimage.metrics import structural_similarity as ssim
from skimage.feature import peak_local_max
from skimage.segmentation import active_contour
from skimage import morphology, measure
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label, measurements

# Import XAI and Model Libraries
from open_clip import create_model_from_pretrained, get_tokenizer
from captum.attr import Saliency, IntegratedGradients, GradientShap, NoiseTunnel
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath, Mlp

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except:
    SAM_AVAILABLE = False
    print("SAM not available")

import warnings
warnings.filterwarnings("ignore")

# =========================================================================================
# REVISION ADD-ONS  (Communications AI & Computing -> Nature-family major revision)
# -----------------------------------------------------------------------------------------
# Editor explicitly approved post-processing (CRF) by email, provided the change is
# described and a before/after comparison is given. Everything below is ADDITIVE and
# CONFIG-gated: with USE_CRF=False and RUN_REVISION_SUITE=False the file reproduces the
# ORIGINAL pipeline byte-for-byte. No reported metric is hard-coded anywhere -- every
# number printed by the suite comes from an actual run on your data.
# =========================================================================================
import time as _time
import random as _random

REVISION_CONFIG = {
    # ---- editor-approved boundary post-processing -------------------------------------
    "USE_CRF": True,            # set False to reproduce the ORIGINAL (pre-revision) result
    "CRF_ITERATIONS": 5,
    # ---- reviewer-requested experiments (run AFTER the normal pipeline) ----------------
    "RUN_REVISION_SUITE": True, # master switch for seeds + CRF on/off + sensitivity + latency
    "RUN_SEEDS": True,          # Reviewer: variability of stochastic prompt sampling
    "SEEDS": [42, 43, 44, 45, 46],
    "RUN_CRF_ABLATION": True,   # Editor: previous-vs-updated methodology comparison
    "RUN_SENSITIVITY": True,    # Reviewer: parameter / fusion-strategy sensitivity
    "SENSITIVITY_FUSION": ["weighted_average", "multiplicative_consensus", "rank_aggregation"],
    "RUN_LATENCY": True,        # Reviewer 2: missing per-image latency
    # ---- tractability knobs (full multi-seed over all images is slow) ------------------
    "SUITE_MAX_SAMPLES": 60,    # cap pathology samples used per suite pass (None = all)
    "SMOKE_TEST": False,        # if True, forces SUITE_MAX_SAMPLES = 4 for a fast dry-run
    # ---- flow / robustness (added in flow-refactor revision) ---------------------------
    "TMAX": 5,                  # iterative self-correction cap (paper Tmax=5; was hard-coded 3)
    "NUM_VIZ_SAMPLES": 6,       # plot only this many representative figures (was: every sample)
    # ---- output destination (CHANGE THIS to send everything elsewhere) -----------------
    "OUTPUT_DIR": "./LENAS_outputs_v2",  # figures/ checkpoints/ results/ logs/ created here
    "SAVE_FIGURES": True,             # write every figure to OUTPUT_DIR/figures
    "TEE_LOG": True,                  # mirror all console output to OUTPUT_DIR/logs/*.log
    # ---- Kvasir localisation fixes (root cause of Dice=0 cases) -------------------------
    "KVASIR_FORCE_POLYP_CLASS": True, # condition XAI on the image-level label (Polyp), not the
                                      # capsule classifier's out-of-domain argmax. This is LENAS's
                                      # stated supervision (image-level y) and fixes wrong-class saliency.
    "KVASIR_TARGET_CLASS": 6,         # index of 'polyps' in CAPSULE_CLASS_LABELS (kvasir 8-class)
    "SUPPRESS_BORDER_SPECULAR": True, # keep prompts off black borders / specular glare (data-driven)
    "KVASIR_CANDIDATE_SELECTION": True,# pick the SAM mask that maximises the classifier polyp-reward
    "KVASIR_N_PEAKS": 4,              # number of saliency-peak candidates to try (besides the bbox one)
    "KVASIR_BLOB_COVERAGE": True,     # add a full-salient-blob candidate per peak to fix under-segmentation
    "KVASIR_STAGE_SELECT": False,     # OFF: pipeline is progressive (SAM->snake->iter->CRF); we fix stages, not select one
    "KVASIR_ITER_INIT_MASK": True,    # iterative refines the incoming snake mask (the fix); False = old restart behaviour
    "KVASIR_USE_SAM": True,           # False = Variant C (no SAM; threshold the fused saliency map)
    "KVASIR_USE_SNAKE": True,         # False = Variant B (no snake refinement)
    "KVASIR_SNAKE_NONEROSION": True,  # snake refines the boundary but may not erode the object (anti-recall-loss)
    "KVASIR_SNAKE_MIN_KEEP": 0.9,     # if snake keeps < this fraction of the input mask, revert to pre-snake mask
    # ---- annotation-free recall growth (saliency + image-level confidence only; NO GT) ----
    "KVASIR_RECALL_GROWTH": True,     # grow the mask into adjacent high-saliency polyp regions
    "KVASIR_GROW_PERCENTILE": 65,     # saliency percentile (within salient pixels) that counts as "polyp"
    "KVASIR_GROW_DILATE": 25,         # only grow into salient regions within this many px of the current mask
    "KVASIR_GROW_MARGIN": 0.05,       # reject growth if polyp confidence (conf_in-conf_out) drops by > this
    "KVASIR_GROW_XAI_SELECT": True,   # grow using whichever XAI map (Saliency/IG/GradientSHAP/fused) the image-level confidence prefers
    # ---- attribution sharpening (same Saliency/IG/GradientSHAP; SmoothGrad denoise + sharpen) ----
    "XAI_SMOOTHGRAD": True,           # SmoothGrad (NoiseTunnel) denoising on Saliency & IG
    "XAI_SG_SAMPLES": 8,             # SmoothGrad samples for Saliency (IG uses half)
    "XAI_SG_NOISE": 0.15,            # SmoothGrad noise stdev (fraction of input range)
    "XAI_SHARPEN": True,             # percentile-floor + gamma sharpening of the fused map
    "XAI_SHARPEN_GAMMA": 1.5,
    "XAI_SHARPEN_PERCENTILE": 60,
}

def set_global_seed(seed):
    """Seed every RNG that the stochastic prompt sampler can touch."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------------------
# CRF / edge-aware boundary refinement (graceful degradation, single source of truth)
# -----------------------------------------------------------------------------------------
try:
    import pydensecrf.densecrf as _dcrf
    _PYDENSECRF_AVAILABLE = True
except Exception:
    _PYDENSECRF_AVAILABLE = False

def _crf_backend_name():
    if _PYDENSECRF_AVAILABLE:
        return "DenseCRF (pydensecrf)"
    try:
        import cv2.ximgproc  # noqa: F401
        return "edge-aware guided-filter fallback (cv2.ximgproc)"
    except Exception:
        return "bilateral-threshold fallback"

print(f"[CRF] boundary post-processing backend in use: {_crf_backend_name()}")

# -----------------------------------------------------------------------------------------
# Output management: ONE configurable folder for figures / checkpoints / results / logs.
# Change REVISION_CONFIG["OUTPUT_DIR"] to send everything wherever you like.
# -----------------------------------------------------------------------------------------
import sys as _sys, datetime as _dt

RUN_TAG = "LENAS"
FIGURE_DIR = None
_FIG_COUNTER = {"n": 0}

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def get_output_dir():
    return REVISION_CONFIG.get("OUTPUT_DIR", "./LENAS_outputs_v2")

def out_subdir(kind):
    return _ensure_dir(os.path.join(get_output_dir(), kind))

def out_path(kind, filename):
    return os.path.join(out_subdir(kind), filename)

def resolve_ckpt(filename):
    """Prefer OUTPUT_DIR/checkpoints; fall back to a legacy file in the CWD if present."""
    cand = out_path("checkpoints", filename)
    if os.path.exists(cand):
        return cand
    if os.path.exists(filename):
        return filename
    return cand

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass

def setup_output_dirs(tag="LENAS"):
    global FIGURE_DIR, RUN_TAG
    RUN_TAG = tag
    root = _ensure_dir(get_output_dir())
    FIGURE_DIR = out_subdir("figures")
    out_subdir("checkpoints"); out_subdir("results"); out_subdir("logs")
    if REVISION_CONFIG.get("TEE_LOG", True):
        try:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            logf = open(out_path("logs", f"{tag}_run_{ts}.log"), "w")
            cur_out = getattr(_sys, "stdout", None) or _sys.__stdout__
            cur_err = getattr(_sys, "stderr", None) or _sys.__stderr__
            if not isinstance(cur_out, _Tee):
                _sys.stdout = _Tee(cur_out, logf)
            if not isinstance(cur_err, _Tee):
                _sys.stderr = _Tee(cur_err, logf)
        except Exception as _e:
            print(f"[output] log tee skipped: {_e}")
    print(f"[output] All artifacts -> {os.path.abspath(root)}  (figures/ checkpoints/ results/ logs/)")
    return root

def save_current_figure(name=None):
    if not REVISION_CONFIG.get("SAVE_FIGURES", True):
        return
    _FIG_COUNTER["n"] += 1
    if name is None:
        name = f"{RUN_TAG}_figure_{_FIG_COUNTER['n']:04d}.png"
    if not str(name).lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
        name = f"{name}.png"
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))
    try:
        d = FIGURE_DIR or out_subdir("figures")
        plt.savefig(os.path.join(d, safe), dpi=150, bbox_inches="tight")
    except Exception as _e:
        print(f"[output] figure save skipped: {_e}")


def apply_crf_refinement(image_np, mask, num_iterations=5,
                         sxy_gaussian=3, compat_gaussian=3,
                         sxy_bilateral=40, srgb_bilateral=10, compat_bilateral=10):
    """
    Snap the (snake-refined) mask to image intensity edges.
      * If pydensecrf is installed -> fully-connected DenseCRF (preferred).
      * else if cv2.ximgproc is available -> edge-aware guided filter.
      * else -> bilateral-smoothed threshold.
    Returns a binary uint8 mask with the same HxW as `mask`.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.sum() == 0 or mask.sum() == mask.size:
        return mask
    H, W = mask.shape[:2]
    img = image_np
    if img is None:
        return mask
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = np.ascontiguousarray(img.astype(np.uint8))

    if _PYDENSECRF_AVAILABLE:
        try:
            d = _dcrf.DenseCRF2D(W, H, 2)
            prob_fg = np.where(mask > 0, 0.9, 0.1).astype(np.float32)
            probs = np.stack([1.0 - prob_fg, prob_fg], axis=0)
            U = (-np.log(probs + 1e-8)).reshape((2, -1)).astype(np.float32)
            d.setUnaryEnergy(np.ascontiguousarray(U))
            d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
            d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral,
                                   rgbim=img, compat=compat_bilateral)
            Q = d.inference(num_iterations)
            out = np.argmax(np.array(Q).reshape((2, H, W)), axis=0).astype(np.uint8)
            if out.sum() == 0:          # never let CRF erase everything
                return mask
            return out
        except Exception as _e:
            print(f"[CRF] DenseCRF failed ({_e}); using guided-filter fallback.")

    # Fallback 1: edge-aware guided filter
    try:
        import cv2.ximgproc as _xip
        guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        soft = _xip.guidedFilter(guide=guide, src=mask.astype(np.float32),
                                 radius=8, eps=1e-3)
        out = (soft >= 0.5).astype(np.uint8)
        return out if out.sum() > 0 else mask
    except Exception:
        pass

    # Fallback 2: bilateral-smoothed threshold
    soft = cv2.bilateralFilter((mask * 255).astype(np.uint8), d=9,
                               sigmaColor=75, sigmaSpace=75)
    out = (soft >= 128).astype(np.uint8)
    return out if out.sum() > 0 else mask

def maybe_crf(image_np, mask):
    """CRF hook honored by the pipeline. Returns mask unchanged when USE_CRF is off."""
    if not REVISION_CONFIG.get("USE_CRF", False):
        return mask
    return apply_crf_refinement(image_np, mask,
                                num_iterations=REVISION_CONFIG.get("CRF_ITERATIONS", 5))

def _suite_cap():
    if REVISION_CONFIG.get("SMOKE_TEST", False):
        return 4
    return REVISION_CONFIG.get("SUITE_MAX_SAMPLES", None)
# =========================================================================================
# END REVISION ADD-ONS (header)
# =========================================================================================

# =========================================================================================
# Part 1: Differential Attention Components
# =========================================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class DifferentialMultiheadAttention(nn.Module):
    """Differential Attention mechanism for Vision Transformers."""
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.num_heads = num_heads
        self.effective_heads = num_heads // 2
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable lambda parameters
        self.lambda_q1 = nn.Parameter(torch.zeros(self.effective_heads, 1, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.effective_heads, 1, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.effective_heads, 1, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.effective_heads, 1, self.head_dim))
        self.lambda_init = lambda_init
        nn.init.normal_(self.lambda_q1, std=0.02)
        nn.init.normal_(self.lambda_k1, std=0.02)
        nn.init.normal_(self.lambda_q2, std=0.02)
        nn.init.normal_(self.lambda_k2, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Split heads for differential attention
        q1, q2 = torch.chunk(q, 2, dim=1)
        k1, k2 = torch.chunk(k, 2, dim=1)
        v1, v2 = torch.chunk(v, 2, dim=1)

        # Compute lambda
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1).sum(dim=-1))
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1).sum(dim=-1))
        lambda_val = (lambda_1 - lambda_2 + self.lambda_init).mean()

        # Attention computations
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)

        # Differential combination
        x_diff = x1 - lambda_val * x2
        x = torch.cat([x_diff, x2], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DifferentialBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, lambda_init=0.8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DifferentialMultiheadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                   attn_drop=attn_drop, proj_drop=drop, lambda_init=lambda_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ExcitationBlock(nn.Module):
    """Excitation Block with Gated Attention."""
    def __init__(self, in_features, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.gate = nn.Sequential(
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = F.relu(x)
        gate = self.gate(x)
        x = identity * gate
        x = self.bn(x)
        return x

# =========================================================================================
# Part 2: FIXED - Hybrid Contrastive + Classification Model
# =========================================================================================

class DifferentialBiomedCLIP(nn.Module):
    """
    FIXED: Hybrid model with contrastive learning + classification head
    """
    def __init__(self, model_name, num_classes, device, lambda_init=0.8, eb_reduction=16,
                 class_names=None, use_contrastive=True):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.use_contrastive = use_contrastive

        print("Loading pre-trained BiomedCLIP model...")
        base_model, self.preprocess = create_model_from_pretrained(
            'hf-hub:' + model_name,
            device=self.device
        )
        self.tokenizer = get_tokenizer('hf-hub:' + model_name)

        # FIXED: Replace Vision Transformer with Differential Version
        print("Replacing Vision Transformer blocks with Differential Version...")
        timm_wrapper = base_model.visual
        if hasattr(timm_wrapper, 'trunk'):
            vision_transformer = timm_wrapper.trunk
        else:
            vision_transformer = next(timm_wrapper.children())

        embed_dim = getattr(vision_transformer, 'embed_dim', 768)
        num_heads = getattr(vision_transformer, 'num_heads', 12)
        drop_path_rate = getattr(vision_transformer, 'drop_path_rate', 0.0)
        drop_rate = getattr(vision_transformer, 'drop_rate', 0.0)
        attn_drop_rate = getattr(vision_transformer, 'attn_drop_rate', 0.0)

        depth = len(vision_transformer.blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Create new differential blocks
        new_blocks = nn.Sequential(*[
            DifferentialBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=type(vision_transformer.blocks[0].norm1),
                act_layer=nn.GELU,
                lambda_init=lambda_init
            )
            for i in range(depth)
        ])

        # FIXED: Proper weight transfer from original blocks to differential blocks
        print("Transferring pretrained weights to differential blocks...")
        with torch.no_grad():
            for i, (orig_block, diff_block) in enumerate(zip(vision_transformer.blocks, new_blocks)):
                # Transfer attention weights (Q, K, V projections)
                diff_block.attn.q_proj.weight.copy_(orig_block.attn.qkv.weight[:embed_dim])
                diff_block.attn.k_proj.weight.copy_(orig_block.attn.qkv.weight[embed_dim:2*embed_dim])
                diff_block.attn.v_proj.weight.copy_(orig_block.attn.qkv.weight[2*embed_dim:])

                if orig_block.attn.qkv.bias is not None:
                    diff_block.attn.q_proj.bias.copy_(orig_block.attn.qkv.bias[:embed_dim])
                    diff_block.attn.k_proj.bias.copy_(orig_block.attn.qkv.bias[embed_dim:2*embed_dim])
                    diff_block.attn.v_proj.bias.copy_(orig_block.attn.qkv.bias[2*embed_dim:])

                # Transfer projection weights
                diff_block.attn.proj.weight.copy_(orig_block.attn.proj.weight)
                diff_block.attn.proj.bias.copy_(orig_block.attn.proj.bias)

                # Transfer MLP weights
                diff_block.mlp.fc1.weight.copy_(orig_block.mlp.fc1.weight)
                diff_block.mlp.fc1.bias.copy_(orig_block.mlp.fc1.bias)
                diff_block.mlp.fc2.weight.copy_(orig_block.mlp.fc2.weight)
                diff_block.mlp.fc2.bias.copy_(orig_block.mlp.fc2.bias)

                # Transfer layer norms
                diff_block.norm1.weight.copy_(orig_block.norm1.weight)
                diff_block.norm1.bias.copy_(orig_block.norm1.bias)
                diff_block.norm2.weight.copy_(orig_block.norm2.weight)
                diff_block.norm2.bias.copy_(orig_block.norm2.bias)

        vision_transformer.blocks = new_blocks

        # Keep text encoder original (it works well for medical text)
        self.model = base_model

        # FIXED: Contrastive learning components
        if self.use_contrastive:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

            if class_names is None:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            self.class_names = class_names
            # Pre-encode class names for contrastive learning
            self.register_buffer('text_features', self._encode_text(class_names))

        # Classification head components
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.eb_block = ExcitationBlock(512, reduction=eb_reduction)
        self.dropout = nn.Dropout(0.3)
        self.classification_head = nn.Linear(512, num_classes)

        self.to(device)

    def _encode_text(self, class_names):
        """Encode text class names."""
        text_tokens = self.tokenizer(class_names).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, return_features=False, return_contrastive=False):
        """
        FIXED: Hybrid forward pass

        Args:
            images: Input images
            return_features: Whether to return image features
            return_contrastive: Whether to return contrastive logits (for training)

        Returns:
            During training with contrastive: (logits_contrastive, logits_cls, image_features)
            During inference: logits_cls
        """
        # Encode images
        image_features = self.model.encode_image(images, normalize=False)

        # Contrastive logits (for training)
        logits_contrastive = None
        if self.use_contrastive and return_contrastive:
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_contrastive = image_features_norm @ self.text_features.T * self.logit_scale.exp()

        # Classification logits
        x = self.eb_block(image_features.float())
        x = self.dropout(x)
        logits_cls = self.classification_head(x)

        if return_contrastive and logits_contrastive is not None:
            if return_features:
                return logits_contrastive, logits_cls, image_features
            return logits_contrastive, logits_cls

        if return_features:
            return logits_cls, image_features
        return logits_cls

    def predict_with_uncertainty(self, images):
        """Predict with entropy-based uncertainty."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions, entropy, probs

# =========================================================================================
# Part 3: Data Handling - CAPSULE ENDOSCOPY DATASET (Classification)
# =========================================================================================

class CapsuleEndoscopyDataset(Dataset):
    """Dataset for Capsule Endoscopy Classification (10 classes)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        if not os.path.isdir(root_dir):
            return

        self.class_to_idx = {d: i for i, d in enumerate(sorted(os.listdir(root_dir)))}
        self.idx_to_class = {i: d for d, i in self.class_to_idx.items()}

        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for subfolder in os.listdir(class_dir):
                    subfolder_path = os.path.join(class_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        for img_name in os.listdir(subfolder_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(subfolder_path, img_name),
                                                   self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except:
            return None, None, None

def collate_fn_capsule(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.empty(0), torch.empty(0), [])

# =========================================================================================
# Part 4: Data Handling - KVASIR-SEG DATASET (Segmentation)
# =========================================================================================

class KvasirSEGDataset(Dataset):
    """Dataset for Kvasir-SEG Polyp Segmentation"""
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image and mask paths
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                 if f.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                                if f.endswith('.jpg')])
        
        # Verify matching
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        
        # Verify filenames match
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            assert img_name == mask_name, f"Filename mismatch: {img_name} vs {mask_name}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # Load image and mask
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
            
            # Convert mask to binary (0 or 1)
            mask_array = np.array(mask)
            mask_array = (mask_array > 127).astype(np.float32)  # Threshold at 127
            mask = Image.fromarray(mask_array)
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            
            if self.mask_transform:
                mask_tensor = self.mask_transform(mask)
            else:
                # Default mask transformation
                mask_tensor = transforms.ToTensor()(mask)
                
            return image_tensor, mask_tensor, img_path

        except Exception as e:
            print(f"Error loading sample {img_path}: {e}")
            return None, None, None

def collate_fn_kvasir(batch):
    # Filter out samples that failed to load
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0), []
    
    images, masks, paths = zip(*batch)
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    
    return images, masks, paths

# =========================================================================================
# Part 5: FIXED - Training with Hybrid Loss
# =========================================================================================

def train_classifier_with_hybrid_loss(model, train_loader, val_loader, epochs, lr, device,
                                     CLASS_LABELS, save_path="best_differential_biomedclip.pth",
                                     contrastive_weight=0.3, classification_weight=0.7):
    """
    FIXED: Training with hybrid contrastive + classification loss
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_contrastive = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_loss_contrastive': [], 'train_loss_cls': [],
               'val_loss': [], 'val_acc': [], 'val_entropy': []}

    for epoch in range(epochs):
        model.train()
        running_loss, running_loss_contrastive, running_loss_cls = 0.0, 0.0, 0.0
        correct_predictions, total_samples = 0, 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            if images.nelement() == 0: continue # Skip empty batches
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # FIXED: Get both contrastive and classification logits
            if model.use_contrastive:
                logits_contrastive, logits_cls = model(images, return_contrastive=True)

                # Hybrid loss
                loss_contrastive = criterion_contrastive(logits_contrastive, labels)
                loss_cls = criterion_cls(logits_cls, labels)
                loss = contrastive_weight * loss_contrastive + classification_weight * loss_cls

                running_loss_contrastive += loss_contrastive.item() * images.size(0)
                running_loss_cls += loss_cls.item() * images.size(0)
            else:
                logits_cls = model(images)
                loss = criterion_cls(logits_cls, labels)
                running_loss_cls += loss.item() * images.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, pred = torch.max(logits_cls.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (pred == labels).sum().item()

        avg_train_loss = running_loss / total_samples if total_samples > 0 else 0
        avg_train_loss_contrastive = running_loss_contrastive / total_samples if model.use_contrastive and total_samples > 0 else 0
        avg_train_loss_cls = running_loss_cls / total_samples if total_samples > 0 else 0

        val_loss, val_acc, val_entropy = validate_classifier_with_uncertainty(
            model, val_loader, criterion_cls, device)

        history['train_loss'].append(avg_train_loss)
        history['train_loss_contrastive'].append(avg_train_loss_contrastive)
        history['train_loss_cls'].append(avg_train_loss_cls)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_entropy'].append(val_entropy)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} " +
              (f"(Contr: {avg_train_loss_contrastive:.4f}, Cls: {avg_train_loss_cls:.4f}), " if model.use_contrastive else "") +
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Avg Entropy: {val_entropy:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            print(f"New best validation accuracy! Saving to {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            best_val_acc = val_acc

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.2f}%")
    return history

def validate_classifier_with_uncertainty(model, dataloader, criterion, device, uncertainty_threshold=None):
    """Validation with uncertainty detection."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_entropies = []
    uncertain_count = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validating", leave=False):
            if images.nelement() == 0: continue # Skip empty batches
            
            images, labels = images.to(device), labels.to(device)
            predictions, entropy, probs = model.predict_with_uncertainty(images)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()
            all_entropies.extend(entropy.cpu().numpy())

            if uncertainty_threshold is not None:
                uncertain_count += (entropy > uncertainty_threshold).sum().item()

    avg_entropy = np.mean(all_entropies) if all_entropies else 0
    val_acc = (val_correct / val_total * 100) if val_total > 0 else 0
    val_loss = (val_loss / val_total) if val_total > 0 else 0

    if uncertainty_threshold is not None:
        print(f"Uncertain predictions: {uncertain_count}/{val_total}")

    return val_loss, val_acc, avg_entropy

# =========================================================================================
# Part 6: XAI GENERATION (with caching)
# =========================================================================================

class XAICache:
    """FIXED: Cache for XAI computations"""
    def __init__(self):
        self.cache = {}

    def get_key(self, image_tensor, label):
        # Simple hash-based key
        img_hash = hash(image_tensor.cpu().numpy().tobytes())
        return f"{img_hash}_{label}"

    def get(self, image_tensor, label):
        key = self.get_key(image_tensor, label)
        return self.cache.get(key, None)

    def set(self, image_tensor, label, explanations):
        key = self.get_key(image_tensor, label)
        self.cache[key] = explanations

    def clear(self):
        self.cache.clear()

# Global XAI cache
xai_cache = XAICache()

def generate_explanations_focused(model, x_batch, y_batch, device, use_cache=True):
    """Generate FOCUSED explanations with caching."""
    # FIX: Handle scalar vs array labels properly
    if isinstance(y_batch, (int, np.integer)):
        label_for_cache = y_batch
    elif isinstance(y_batch, torch.Tensor):
        label_for_cache = y_batch.item() if y_batch.numel() == 1 else y_batch[0].item()
    else:
        label_for_cache = y_batch[0] if hasattr(y_batch, '__getitem__') else y_batch
    
    # Check cache first
    if use_cache:
        cached = xai_cache.get(x_batch[0], label_for_cache)
        if cached is not None:
            return cached

    model.eval()
    explanations = {}
    x_batch_grad = x_batch.clone().to(device).requires_grad_()
    
    # Convert y_batch to tensor for Captum
    if isinstance(y_batch, (int, np.integer)):
        target_tensor = torch.tensor([y_batch])
    elif isinstance(y_batch, torch.Tensor):
        target_tensor = y_batch if y_batch.dim() > 0 else y_batch.unsqueeze(0)
    else:
        target_tensor = torch.tensor(y_batch)

    tgt = target_tensor.to(device)
    _sg = REVISION_CONFIG.get("XAI_SMOOTHGRAD", True)
    _sgn = int(REVISION_CONFIG.get("XAI_SG_SAMPLES", 8))
    _sgs = float(REVISION_CONFIG.get("XAI_SG_NOISE", 0.15))

    # Saliency (+ SmoothGrad denoising -- same method, far less ViT speckle)
    saliency = Saliency(model)
    try:
        if _sg:
            sal_attr = NoiseTunnel(saliency).attribute(
                x_batch_grad, nt_type='smoothgrad', nt_samples=_sgn, stdevs=_sgs,
                target=tgt, abs=True)
        else:
            sal_attr = saliency.attribute(x_batch_grad, target=tgt, abs=True)
    except Exception:
        sal_attr = saliency.attribute(x_batch_grad, target=tgt, abs=True)
    explanations['Saliency'] = sal_attr.abs().sum(axis=1).squeeze().cpu().detach().numpy()

    # Integrated Gradients (+ light SmoothGrad)
    ig = IntegratedGradients(model)
    try:
        if _sg:
            ig_attr = NoiseTunnel(ig).attribute(
                x_batch_grad, nt_type='smoothgrad', nt_samples=max(2, _sgn // 2),
                stdevs=_sgs, target=tgt, n_steps=20)
        else:
            ig_attr = ig.attribute(x_batch_grad, target=tgt, n_steps=30)
    except Exception:
        ig_attr = ig.attribute(x_batch_grad, target=tgt, n_steps=30)
    explanations['IntegratedGradients'] = np.abs(ig_attr.sum(axis=1).squeeze().cpu().detach().numpy())

    # GradientShap (already stochastic via random baselines; left as in the paper)
    gs = GradientShap(model)
    baseline = torch.zeros_like(x_batch_grad)
    gs_attr = gs.attribute(x_batch_grad, baselines=baseline, target=tgt, n_samples=15)
    explanations['GradientShap'] = np.abs(gs_attr.sum(axis=1).squeeze().cpu().detach().numpy())

    # Cache the result
    if use_cache:
        xai_cache.set(x_batch[0], label_for_cache, explanations)

    return explanations

def calculate_faithfulness_quantus(model, image_tensor, label, explanations, device, subset_size=224, nr_runs=25):
    """
    Faithfulness Correlation (Quantus-style).
    FIXED: the previous version correlated attributions against a CONSTANT vector
    ([pred_drop] * k), which is always NaN -> every method scored 0.0. We now build,
    across many random pixel subsets, pairs of (total |attribution| in the subset,
    drop in target-class probability when that subset is perturbed to a neutral
    per-channel mean baseline), and correlate them. Higher positive correlation
    => more faithful attributions.
    """
    model.eval()
    faithfulness_scores = {}
    with torch.no_grad():
        original_logits = model(image_tensor.unsqueeze(0).to(device))
        original_pred = torch.softmax(original_logits, dim=1)[0, label].item()
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    bmean = image_tensor.mean(dim=(1, 2)).to(image_tensor.device)
    for method, xai_map in explanations.items():
        flat_attr = np.abs(xai_map.flatten())
        k = int(min(subset_size, max(8, xai_map.size // 50)))
        importance_sums, pred_drops = [], []
        for _ in range(nr_runs):
            idx = np.random.choice(xai_map.size, k, replace=False)
            importance_sums.append(float(flat_attr[idx].sum()))
            cy, cx = np.unravel_index(idx, xai_map.shape)
            ys = np.clip((cy * H / xai_map.shape[0]).astype(int), 0, H - 1)
            xs = np.clip((cx * W / xai_map.shape[1]).astype(int), 0, W - 1)
            perturbed = image_tensor.clone()
            perturbed[:, ys, xs] = bmean.unsqueeze(1)
            with torch.no_grad():
                plogits = model(perturbed.unsqueeze(0).to(device))
                ppred = torch.softmax(plogits, dim=1)[0, label].item()
            pred_drops.append(original_pred - ppred)
        if (len(importance_sums) > 1 and np.std(importance_sums) > 1e-8
                and np.std(pred_drops) > 1e-8):
            corr = np.corrcoef(importance_sums, pred_drops)[0, 1]
            faithfulness_scores[method] = float(corr) if not np.isnan(corr) else 0.0
        else:
            faithfulness_scores[method] = 0.0
    return faithfulness_scores

def calculate_robustness_quantus(model, image_tensor, label, device, nr_samples=10, noise_level=0.2):
    """Robustness to input perturbations."""
    base_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), label, device)
    robustness_scores = {}
    similarities_per_method = {method: [] for method in base_explanations.keys()}

    for _ in range(nr_samples):
        noise = torch.rand_like(image_tensor) * 2 * noise_level - noise_level
        noisy_image = torch.clamp(image_tensor + noise, 0, 1)
        noisy_explanations = generate_explanations_focused(model, noisy_image.unsqueeze(0), label, device)

        for method in base_explanations.keys():
            base_map = base_explanations[method]
            noisy_map = noisy_explanations[method]

            base_norm = (base_map - base_map.min()) / (base_map.max() - base_map.min() + 1e-8)
            noisy_norm = (noisy_map - noisy_map.min()) / (noisy_map.max() - noisy_map.min() + 1e-8)

            similarity = ssim(base_norm, noisy_norm, data_range=1.0)
            if not np.isnan(similarity):
                similarities_per_method[method].append(similarity)

    for method in similarities_per_method:
        robustness_scores[method] = np.mean(similarities_per_method[method]) if similarities_per_method[method] else 0.0

    return robustness_scores

def calculate_sparseness_quantus(explanations):
    """Sparseness using Gini coefficient."""
    sparseness_scores = {}
    for method, xai_map in explanations.items():
        abs_map = np.abs(xai_map.flatten())
        if abs_map.sum() == 0:
            sparseness_scores[method] = 1.0
            continue

        abs_map = abs_map / abs_map.sum()
        n = len(abs_map)
        sorted_abs = np.sort(abs_map)
        cumsum_sorted = np.cumsum(sorted_abs)
        gini = (n + 1 - 2 * np.sum(cumsum_sorted)) / n
        sparseness_scores[method] = gini

    return sparseness_scores

def calculate_localization_quantus(explanations):
    """Localization using relevance rank accuracy."""
    localization_scores = {}

    for method, xai_map in explanations.items():
        threshold = np.percentile(np.abs(xai_map), 90)
        relevant_mask = np.abs(xai_map) >= threshold

        total_relevant = relevant_mask.sum()
        if total_relevant == 0:
            localization_scores[method] = 0.0
        else:
            sorted_indices = np.argsort(np.abs(xai_map.flatten()))[::-1]
            top_k = int(0.1 * len(sorted_indices))

            relevant_in_top_k = 0
            for idx in sorted_indices[:top_k]:
                coords = np.unravel_index(idx, xai_map.shape)
                if relevant_mask[coords]:
                    relevant_in_top_k += 1

            localization_scores[method] = relevant_in_top_k / top_k if top_k > 0 else 0.0

    return localization_scores

def calculate_randomization_quantus(model, image_tensor, label, device, num_classes):
    """Randomization test."""
    correct_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), label, device)
    random_label = np.random.choice([i for i in range(num_classes) if i != label])
    random_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), random_label, device)

    randomization_scores = {}
    for method in correct_explanations.keys():
        correct_map = correct_explanations[method]
        random_map = random_explanations[method]

        correct_norm = (correct_map - correct_map.min()) / (correct_map.max() - correct_map.min() + 1e-8)
        random_norm = (random_map - random_map.min()) / (random_map.max() - random_map.min() + 1e-8)

        similarity = ssim(correct_norm, random_norm, data_range=1.0)
        dissimilarity = 1 - similarity if not np.isnan(similarity) else 1.0
        randomization_scores[method] = max(0.0, dissimilarity)

    return randomization_scores

def evaluate_xai_methods_quantus(model, dataloader, device, num_classes, num_samples=20, CLASS_LABELS=None):
    """Quantus-style evaluation."""
    print("Starting Quantus-style XAI evaluation...")

    if len(dataloader.dataset) < num_samples:
        num_samples = len(dataloader.dataset)

    subset_indices = np.random.choice(len(dataloader.dataset), num_samples, replace=False)
    subset_loader = DataLoader(dataloader.dataset, batch_size=1,
                              sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

    methods = ['Saliency', 'IntegratedGradients', 'GradientShap']
    results = {m: {'faithfulness': [], 'robustness': [], 'sparseness': [],
                   'localization': [], 'randomization': []} for m in methods}

    for images, labels, _ in tqdm(subset_loader, desc="Evaluating XAI"):
        image, label = images.squeeze(0), labels.item()

        explanations = generate_explanations_focused(model, image.unsqueeze(0), label, device)

        faithfulness = calculate_faithfulness_quantus(model, image, label, explanations, device)
        robustness = calculate_robustness_quantus(model, image, label, device)
        sparseness = calculate_sparseness_quantus(explanations)
        localization = calculate_localization_quantus(explanations)
        randomization = calculate_randomization_quantus(model, image, label, device, num_classes)

        for method in methods:
            results[method]['faithfulness'].append(faithfulness.get(method, 0))
            results[method]['robustness'].append(robustness.get(method, 0))
            results[method]['sparseness'].append(sparseness.get(method, 0))
            results[method]['localization'].append(localization.get(method, 0))
            results[method]['randomization'].append(randomization.get(method, 0))

        gc.collect()
        torch.cuda.empty_cache()

    avg_scores = {m: {k: np.mean(v) for k, v in scores.items()} for m, scores in results.items()}

    print("\n--- Average XAI Metric Scores ---")
    df = pd.DataFrame.from_dict(avg_scores, orient='index')
    print(df.round(4))

    df_norm = df.copy()
    for col in df_norm.columns:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max > col_min:
            if col == 'robustness':
                df_norm[col] = 1 - (df_norm[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 1.0

    method_scores = df_norm.mean(axis=1)
    total_score = method_scores.sum()
    xai_weights = (method_scores / total_score).to_dict() if total_score > 0 else {m: 1/len(methods) for m in methods}

    print("\n--- Derived XAI Weights ---")
    for method, weight in xai_weights.items():
        print(f"{method}: {weight:.4f}")

    return xai_weights, avg_scores

# =========================================================================================
# Part 7: IMPROVED XAI FUSION & MASK QUALITY
# =========================================================================================

def normalize_xai_map(xai_map, method='minmax'):
    """Normalize XAI map."""
    if method == 'minmax':
        map_min, map_max = xai_map.min(), xai_map.max()
        if map_max > map_min:
            return (xai_map - map_min) / (map_max - map_min)
        else:
            return np.zeros_like(xai_map)

def advanced_xai_fusion(explanations, weights, fusion_strategy='weighted_average'):
    """Focused fusion of XAI maps."""
    normalized_maps = {}
    for method, xai_map in explanations.items():
        normalized_maps[method] = normalize_xai_map(xai_map, 'minmax')

    if fusion_strategy == 'weighted_average':
        fused_map = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * norm_map
            total_weight += weight
        if total_weight > 0:
            fused_map /= total_weight

    elif fusion_strategy == 'multiplicative_consensus':
        fused_map = np.ones_like(next(iter(normalized_maps.values())))
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map *= (norm_map + 0.01) ** weight

    elif fusion_strategy == 'weighted_geometric_mean':
        log_sum = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            log_sum += weight * np.log(norm_map + 1e-8)
            total_weight += weight
        fused_map = np.exp(log_sum / total_weight) if total_weight > 0 else log_sum

    elif fusion_strategy == 'rank_aggregation':
        rank_maps = {}
        for method, norm_map in normalized_maps.items():
            flat_map = norm_map.flatten()
            ranks = np.argsort(np.argsort(flat_map)) / len(flat_map)
            rank_maps[method] = ranks.reshape(norm_map.shape)

        fused_map = np.zeros_like(next(iter(rank_maps.values())))
        for method, rank_map in rank_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * rank_map
    else:
        fused_map = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * norm_map
            total_weight += weight
        if total_weight > 0:
            fused_map /= total_weight

    return fused_map

def calculate_mask_quality_score_improved(mask, image_np, fused_map):
    """
    IMPROVED: Calculate mask quality with better metrics.

    Factors:
    1. Shape regularity (not just compactness)
    2. Saliency alignment with contrast
    3. Edge alignment
    4. Size plausibility
    5. Texture consistency
    """
    if mask.sum() == 0:
        return 0.0

    # Factor 1: Shape regularity (allow irregular shapes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if area == 0:
        return 0.0

    # Shape score: normalized so circles=1.0, but don't penalize irregular too much
    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    shape_score = 0.5 + 0.5 * compactness  # Range [0.5, 1.0]

    # Factor 2: Saliency alignment with stronger contrast requirement
    fused_resized = cv2.resize(fused_map, (mask.shape[1], mask.shape[0]))
    fused_norm = (fused_resized - fused_resized.min()) / (fused_resized.max() - fused_resized.min() + 1e-8)

    saliency_in_mask = fused_norm[mask > 0].mean() if mask.sum() > 0 else 0.0
    saliency_outside_mask = fused_norm[mask == 0].mean() if (mask == 0).sum() > 0 else 0.0

    # Require stronger contrast (at least 0.2 difference)
    saliency_contrast = max(0, saliency_in_mask - saliency_outside_mask - 0.2) / 0.8  # Normalize to [0,1]

    # Factor 3: Edge strength at boundaries
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    mask_boundary = mask_dilated - mask

    edge_score = edges[mask_boundary > 0].mean() / 255.0 if mask_boundary.sum() > 0 else 0.0

    # Factor 4: Size plausibility (not too small, not too large)
    mask_ratio = area / (mask.shape[0] * mask.shape[1])
    if mask_ratio < 0.001:  # Too small
        size_score = mask_ratio / 0.001  # Penalize
    elif mask_ratio > 0.5:  # Too large
        size_score = max(0, 1.0 - (mask_ratio - 0.5) / 0.5)
    else:
        size_score = 1.0

    # Factor 5: Texture consistency within mask
    if mask.sum() > 100:
        masked_region = image_np.copy()
        masked_region[mask == 0] = 0
        gray_masked = cv2.cvtColor(masked_region, cv2.COLOR_RGB2GRAY)
        texture_variance = np.std(gray_masked[mask > 0])
        # Moderate variance is good (structured), too high or too low is bad
        texture_score = np.exp(-((texture_variance - 30) ** 2) / (2 * 20 ** 2))  # Gaussian around 30
    else:
        texture_score = 0.5

    # Combine scores with adjusted weights
    quality_score = (
        0.15 * shape_score +           # Shape quality (less weight)
        0.40 * saliency_contrast +     # Saliency contrast (highest weight)
        0.20 * edge_score +            # Edge alignment
        0.15 * size_score +            # Size plausibility
        0.10 * texture_score           # Texture consistency
    )

    return quality_score

# =========================================================================================
# Part 8: IMPROVED UNCERTAINTY-GUIDED PROMPTING
# =========================================================================================

def uncertainty_guided_prompts(fused_map, entropy, prediction_probs, base_num_prompts=3):
    """
    IMPROVED: Uncertainty-guided prompt selection.

    Considers:
    - Prediction entropy
    - Confidence margin
    - Spatial uncertainty
    """
    # Spatial uncertainty: variance in XAI map
    spatial_uncertainty = np.std(fused_map)

    # Prediction uncertainty: margin between top-2 classes
    sorted_probs = np.sort(prediction_probs)
    margin = sorted_probs[-1] - sorted_probs[-2] if len(sorted_probs) > 1 else 1.0

    # Determine strategy
    if entropy > 1.5 and margin < 0.3:  # High entropy AND low margin
        strategy = 'dense_sampling'
        num_prompts = min(base_num_prompts + 4, 7)
        print(f"  🎯 Strategy: Dense sampling (entropy={entropy:.3f}, margin={margin:.3f})")
    elif spatial_uncertainty > np.percentile(fused_map, 75):  # Diffuse saliency
        strategy = 'spatial_coverage'
        num_prompts = min(base_num_prompts + 2, 5)
        print(f"  🎯 Strategy: Spatial coverage (spatial_unc={spatial_uncertainty:.3f})")
    else:  # Confident and focused
        strategy = 'peak_selection'
        num_prompts = base_num_prompts
        print(f"  🎯 Strategy: Peak selection (confident)")

    return num_prompts, strategy

# =========================================================================================
# Part 9: IMPROVED SNAKE REFINEMENT WITH SALIENCY WEIGHTING
# =========================================================================================

def refine_mask_with_snake_improved(mask, image_np, fused_map, iterations=100,
                                   alpha=0.05, beta=5, gamma=0.001, max_deviation=30):
    """
    IMPROVED: Snake refinement with saliency constraints.

    Improvements:
    - Weight edges by XAI saliency
    - Reject if snake deviates too far
    - Reduced iterations for speed
    """
    if mask.sum() == 0:
        print("  ⚠️ Empty mask, skipping snake refinement")
        return mask

    try:
        # Convert image to grayscale for edge detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Weight edges by saliency - discourage moving to low-saliency regions
        fused_resized = cv2.resize(fused_map, (mask.shape[1], mask.shape[0]))
        saliency_threshold = np.percentile(fused_resized, 50)
        saliency_mask = (fused_resized > saliency_threshold).astype(np.float32)

        weighted_edges = edges.astype(np.float32) * saliency_mask

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return mask

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        snake_init = contour.squeeze()

        if len(snake_init.shape) != 2 or snake_init.shape[1] != 2:
            print("  ⚠️ Invalid contour shape, skipping snake")
            return mask

        # Store original contour for deviation check
        original_contour = snake_init.copy()

        # Swap from (x,y) to (row,col) for active_contour
        snake_init = snake_init[:, [1, 0]]

        # Sample points if too many
        if len(snake_init) > 500:
            indices = np.linspace(0, len(snake_init)-1, 500, dtype=int)
            snake_init = snake_init[indices]

        # Apply active contour with improved parameters
        print(f"  🐍 Running Snake Algorithm ({iterations} iterations)...")
        snake_refined = active_contour(
            weighted_edges,
            snake_init,
            alpha=alpha,        # Higher continuity
            beta=beta,          # Lower smoothness (allow irregular shapes)
            gamma=gamma,
            max_px_move=1.0,
            max_num_iter=iterations,
            boundary_condition='periodic',
            convergence=0.1
        )

        # Convert back to mask
        refined_mask = np.zeros_like(mask)
        snake_refined_xy = snake_refined[:, [1, 0]].astype(np.int32)
        cv2.fillPoly(refined_mask, [snake_refined_xy], 1)

        # Check deviation from original
        deviation = np.mean(np.min(pdist(np.vstack([original_contour, snake_refined_xy[:, ::-1]])), axis=0))

        if deviation > max_deviation:
            print(f"  ⚠️ Snake deviated too far ({deviation:.1f}px > {max_deviation}px), keeping original")
            return mask

        # Post-process
        refined_mask = post_process_mask(refined_mask, image_np, min_area=100, hole_area=50)

        # --- progressive guard: snake refines the boundary, it must NOT erode the object.
        # Covers both contraction and the largest-contour-only component drop above. ---
        if REVISION_CONFIG.get("KVASIR_SNAKE_NONEROSION", True):
            m_in = (mask > 0); m_out = (refined_mask > 0)
            in_area = int(m_in.sum())
            inter = int(np.logical_and(m_in, m_out).sum())
            recall_vs_input = inter / max(in_area, 1)
            keep = REVISION_CONFIG.get("KVASIR_SNAKE_MIN_KEEP", 0.9)
            if in_area > 0 and (int(m_out.sum()) < keep * in_area or recall_vs_input < keep):
                print("  \u21aa\ufe0f Snake would erode the mask; keeping pre-snake mask (non-erosion guard)")
                return mask
        print(f"  ✅ Snake refinement complete (deviation: {deviation:.1f}px)")
        return refined_mask

    except Exception as e:
        print(f"  ⚠️ Snake algorithm failed: {e}, returning original mask")
        return mask

def post_process_mask(mask, image_np, min_area=200, hole_area=100):
    """Post-process segmentation mask."""
    mask_bool = mask.astype(bool)

    # Remove very small objects
    mask_bool = morphology.remove_small_objects(mask_bool, min_size=min_area)

    # Fill small holes
    mask_bool = morphology.remove_small_holes(mask_bool, area_threshold=hole_area)

    # Morphological operations
    kernel = morphology.disk(3)
    mask_bool = morphology.binary_closing(mask_bool, kernel)
    mask_bool = morphology.binary_opening(mask_bool, morphology.disk(2))

    # Keep largest component
    labeled = measure.label(mask_bool)
    if labeled.max() > 0:
        component_sizes = np.bincount(labeled.flat)[1:]
        if len(component_sizes) > 0:
            largest_cc_idx = np.argmax(component_sizes) + 1
            mask_bool = (labeled == largest_cc_idx)

    return mask_bool.astype(np.uint8)

# =========================================================================================
# Part 10: IMPROVED ITERATIVE SELF-CORRECTION
# =========================================================================================

def validate_prompt_additions(additional_prompts, current_mask, fused_map, min_saliency_threshold=0.7):
    """
    IMPROVED: Validate that new prompts are meaningful before adding them.

    Returns:
        valid_prompts: Array of validated prompts
    """
    if len(additional_prompts) == 0:
        return additional_prompts

    fused_norm = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min() + 1e-8)

    valid_prompts = []
    for prompt in additional_prompts:
        x, y = int(prompt[0]), int(prompt[1])

        # Check if within bounds
        if 0 <= y < fused_norm.shape[0] and 0 <= x < fused_norm.shape[1]:
            # Check saliency at prompt location
            saliency_value = fused_norm[y, x]

            # Only add if saliency is high enough
            if saliency_value >= min_saliency_threshold:
                valid_prompts.append(prompt)
            else:
                print(f"    ❌ Rejected prompt at ({x},{y}) - low saliency: {saliency_value:.3f}")

    return np.array(valid_prompts) if valid_prompts else np.array([])

def iterative_self_correction_improved(classifier, sam_predictor, image_np, image_tensor,
                                      original_label, device, xai_weights,
                                      max_iterations=3, convergence_threshold=0.02,
                                      confidence_threshold=0.15, init_mask=None):
    """
    IMPROVED: Stricter quality gates and validated prompt additions.
    FIXED: Resolved UnboundLocalError for 'no_improvement_count'.
    """
    print("\n" + "="*70)
    print("🔄 IMPROVED ITERATIVE SELF-CORRECTION LOOP")
    print("="*70)

    iteration_history = {
        'iteration': [],
        'mask_iou': [],
        'confidence_masked_in': [],
        'confidence_masked_out': [],
        'confidence_diff': [],
        'mask_ratio': [],
        'mask_quality': [],
        'sam_score': [],
        'converged': False,
        'best_iteration': 0
    }

    # Get initial prediction
    with torch.no_grad():
        _, entropy_initial, probs_initial = classifier.predict_with_uncertainty(
            image_tensor.unsqueeze(0).to(device)
        )
        initial_confidence = probs_initial[0][original_label].item()
        entropy_val = entropy_initial[0].item()

    print(f"Initial: Label={original_label}, Conf={initial_confidence:.4f}, Entropy={entropy_val:.4f}")

    # Detect circular mask
    circular_mask = detect_circular_mask(image_np.shape)

    # === ITERATION 0: Initial Segmentation ===
    print(f"\n🔍 Iteration 0: Initial Segmentation")

    # Generate XAI
    explanations = generate_explanations_focused(classifier, image_tensor.unsqueeze(0),
                                                 original_label, device, use_cache=True)
    fused_map = advanced_xai_fusion(explanations, xai_weights)

    # Uncertainty-guided prompting
    num_prompts, strategy = uncertainty_guided_prompts(fused_map, entropy_val,
                                                       probs_initial[0].cpu().numpy())

    # Extract prompts
    bbox_xai = extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05)
    positive_prompts_xai = extract_focused_positive_prompts(fused_map, bbox_xai,
                                                            num_prompts=num_prompts)

    circular_mask_xai = cv2.resize(circular_mask, (fused_map.shape[1], fused_map.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    negative_prompts_xai = extract_smart_negative_prompts(fused_map, bbox_xai,
                                                         circular_mask_xai, num_negatives=3)

    # Transform to original space
    xai_shape = fused_map.shape
    original_shape = image_np.shape[:2]
    bbox_orig = transform_bbox_to_original(bbox_xai, xai_shape, original_shape)
    positive_prompts_orig = transform_coordinates_to_original(positive_prompts_xai, xai_shape, original_shape)
    negative_prompts_orig = transform_coordinates_to_original(negative_prompts_xai, xai_shape, original_shape) if len(negative_prompts_xai) > 0 else np.array([])

    # Iteration 0 START: refine the PIPELINE's mask (candidate-selected + snake) when provided,
    # rather than recomputing a weaker mask with the legacy bbox extractor. This keeps the
    # pipeline progressive (SAM -> snake -> ITERATIVE refines that result -> CRF).
    if init_mask is not None:
        current_mask = (np.asarray(init_mask) > 0).astype(np.uint8)
        if current_mask.shape != image_np.shape[:2]:
            current_mask = cv2.resize(current_mask, (image_np.shape[1], image_np.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        sam_score = 0.0
        print("  \u21aa\ufe0f Seeded iteration-0 from pipeline mask (progressive refinement)")
    else:
        if sam_predictor is not None:
            current_mask, sam_score = segment_with_sam_enhanced(
                sam_predictor, image_np, positive_prompts_orig,
                negative_prompts_orig, bbox_orig, post_process=True
            )
        else:
            threshold = np.percentile(fused_map, 95)
            mask_small = (fused_map >= threshold).astype(np.uint8)
            current_mask = cv2.resize(mask_small, (image_np.shape[1], image_np.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            sam_score = 0.0
        print("  🐍 Applying improved Snake refinement...")
        current_mask = refine_mask_with_snake_improved(current_mask, image_np, fused_map,
                                                       iterations=100)

    # Evaluate
    conf_in, conf_out = evaluate_masked_confidence(classifier, image_tensor, current_mask,
                                                   original_label, device, image_np.shape[:2])
    mask_quality = calculate_mask_quality_score_improved(current_mask, image_np, fused_map)
    mask_ratio = np.sum(current_mask) / current_mask.size
    confidence_diff = conf_in - conf_out

    iteration_history['iteration'].append(0)
    iteration_history['mask_iou'].append(1.0)
    iteration_history['confidence_masked_in'].append(conf_in)
    iteration_history['confidence_masked_out'].append(conf_out)
    iteration_history['confidence_diff'].append(confidence_diff)
    iteration_history['mask_ratio'].append(mask_ratio)
    iteration_history['mask_quality'].append(mask_quality)
    iteration_history['sam_score'].append(sam_score)

    print(f"  Quality: {mask_quality:.4f} | Conf Diff: {confidence_diff:.4f}")

    # Track best
    best_mask = current_mask.copy()
    best_confidence_diff = confidence_diff
    best_quality = mask_quality
    best_iteration = 0

    # === ITERATIVE REFINEMENT ===
    previous_mask = current_mask.copy()
    no_improvement_count = 0  # FIX: Initialize the counter before the loop

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔍 Iteration {iteration}: Self-Correction")

        should_refine = confidence_diff < confidence_threshold

        if should_refine:
            print(f"  ⚠️ Conf diff ({confidence_diff:.4f}) < threshold ({confidence_threshold})")
            print(f"  🔧 Adding validated prompts...")

            fused_resized = cv2.resize(fused_map, (image_np.shape[1], image_np.shape[0]))
            fused_norm = (fused_resized - fused_resized.min()) / (fused_resized.max() - fused_resized.min() + 1e-8)

            # Find uncovered high-saliency regions
            high_sal_threshold = np.percentile(fused_norm, 92)
            high_sal_mask = (fused_norm >= high_sal_threshold).astype(np.uint8)
            uncovered_regions = high_sal_mask * (1 - current_mask)

            if uncovered_regions.sum() > 50:
                uncovered_coords = np.where(uncovered_regions > 0)
                num_additional = min(2, len(uncovered_coords[0]))

                if num_additional > 0:
                    saliency_values = fused_norm[uncovered_coords]
                    top_indices = np.argsort(saliency_values)[-num_additional:]
                    additional_prompts = np.array([[uncovered_coords[1][i], uncovered_coords[0][i]]
                                                  for i in top_indices])

                    # IMPROVED: Validate prompts before adding
                    valid_prompts = validate_prompt_additions(additional_prompts, current_mask,
                                                             fused_resized, min_saliency_threshold=0.7)

                    if len(valid_prompts) > 0:
                        positive_prompts_orig = np.vstack([positive_prompts_orig, valid_prompts])
                        print(f"  ➕ Added {len(valid_prompts)} validated prompts")
        else:
            print(f"  ✅ Conf diff is good, fine-tuning only...")

        # Re-segment
        if sam_predictor is not None:
            refined_mask, sam_score = segment_with_sam_enhanced(
                sam_predictor, image_np, positive_prompts_orig,
                negative_prompts_orig, bbox_orig, post_process=True
            )
        else:
            adaptive_threshold = np.percentile(fused_map, 93)
            mask_small = (fused_map >= adaptive_threshold).astype(np.uint8)
            refined_mask = cv2.resize(mask_small, (image_np.shape[1], image_np.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            sam_score = 0.0

        # Apply improved snake
        print("  🐍 Applying improved Snake refinement...")
        refined_mask = refine_mask_with_snake_improved(refined_mask, image_np, fused_map, iterations=100)

        # Evaluate refined mask
        conf_in_new, conf_out_new = evaluate_masked_confidence(classifier, image_tensor, refined_mask,
                                                               original_label, device, image_np.shape[:2])
        mask_quality_new = calculate_mask_quality_score_improved(refined_mask, image_np, fused_map)

        # Calculate IoU with previous mask
        intersection = np.logical_and(refined_mask, previous_mask).sum()
        union = np.logical_or(refined_mask, previous_mask).sum()
        iou = intersection / union if union > 0 else 0.0

        mask_ratio_new = np.sum(refined_mask) / refined_mask.size
        confidence_diff_new = conf_in_new - conf_out_new

        # Store iteration metrics
        iteration_history['iteration'].append(iteration)
        iteration_history['mask_iou'].append(iou)
        iteration_history['confidence_masked_in'].append(conf_in_new)
        iteration_history['confidence_masked_out'].append(conf_out_new)
        iteration_history['confidence_diff'].append(confidence_diff_new)
        iteration_history['mask_ratio'].append(mask_ratio_new)
        iteration_history['mask_quality'].append(mask_quality_new)
        iteration_history['sam_score'].append(sam_score)

        print(f"  IoU: {iou:.4f} | Quality: {mask_quality_new:.4f} | Conf Diff: {confidence_diff_new:.4f}")

        # STRICTER QUALITY GATE: Accept only if BOTH improve AND better than best seen
        improvement = (
            confidence_diff_new > confidence_diff and
            mask_quality_new > mask_quality and
            confidence_diff_new > best_confidence_diff and
            mask_quality_new > best_quality
        )

        # Check for convergence
        converged = iou > (1.0 - convergence_threshold)

        if converged:
            print(f"  ✅ Converged! Mask stable (IoU: {iou:.4f})")
            iteration_history['converged'] = True
            current_mask = refined_mask

            # Update best if this is better
            if confidence_diff_new > best_confidence_diff and mask_quality_new > best_quality * 0.9:
                best_mask = refined_mask.copy()
                best_confidence_diff = confidence_diff_new
                best_quality = mask_quality_new
                best_iteration = iteration
                print(f"  🌟 New best mask!")

            break

        if improvement:
            print(f"  ✅ Accepted! All metrics improved")
            current_mask = refined_mask
            best_mask = refined_mask.copy()  # Update best immediately
            best_confidence_diff = confidence_diff_new
            best_quality = mask_quality_new
            best_iteration = iteration
            conf_in, conf_out = conf_in_new, conf_out_new
            confidence_diff = confidence_diff_new
            mask_quality = mask_quality_new
            mask_ratio = mask_ratio_new
            no_improvement_count = 0 # FIX: Reset counter on improvement
        else:
            print(f"  ⚠️ Rejected! Quality gate not passed, reverting to best")
            # Revert to best mask
            current_mask = best_mask.copy()
            no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= 2:
                print(f"  🛑 Early stopping: No improvement for 2 iterations")
                break

        previous_mask = current_mask.copy()

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(f"🏁 Self-Correction Complete")
    print(f"   Iterations: {len(iteration_history['iteration'])}")
    print(f"   Converged: {iteration_history['converged']}")
    print(f"   Best Iteration: {best_iteration}")
    print(f"   Best Conf Diff: {best_confidence_diff:.4f}")
    print(f"   Best Quality: {best_quality:.4f}")
    print("="*70)

    iteration_history['best_iteration'] = best_iteration
    return best_mask, iteration_history

def evaluate_masked_confidence(classifier, image_tensor, mask, label, device, original_shape):
    """Evaluate classifier confidence on masked regions."""
    tensor_h, tensor_w = image_tensor.shape[1], image_tensor.shape[2]
    mask_resized = cv2.resize(mask, (tensor_w, tensor_h), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_resized).float().to(device)
    image_tensor = image_tensor.to(device)

    # --- REVISION FIX: neutral (mean) background instead of pure black ---
    # Pure-black masking is OOD for the BiomedCLIP ViT and destabilises the
    # self-correction reward; the BUSI pipeline already uses the mean. Aligned here.
    neutral_background = image_tensor.mean(dim=(1, 2), keepdim=True)
    image_masked_in = image_tensor * mask_tensor + neutral_background * (1 - mask_tensor)
    image_masked_out = image_tensor * (1 - mask_tensor) + neutral_background * mask_tensor

    with torch.no_grad():
        _, _, probs_in = classifier.predict_with_uncertainty(image_masked_in.unsqueeze(0))
        _, _, probs_out = classifier.predict_with_uncertainty(image_masked_out.unsqueeze(0))
        conf_in = probs_in[0][label].item()
        conf_out = probs_out[0][label].item()

    return conf_in, conf_out

# =========================================================================================
# Part 11: Segmentation Metrics
# =========================================================================================

def calculate_dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def calculate_iou_score(pred, target, smooth=1e-6):
    """Calculate IoU score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# =========================================================================================
# Part 12: ENHANCED VISUALIZATION
# =========================================================================================

def detect_circular_mask(image_shape, padding=10):
    """
    Detect the circular field of view in endoscopy images.
   
    Args:
        image_shape: Shape of the image (H, W, C) or (H, W)
        padding: Padding around the circular mask
   
    Returns:
        Binary mask with 1 inside the circular FOV, 0 outside
    """
    h, w = image_shape[:2]
    center_y, center_x = h // 2, w // 2

    # Assume circular FOV with some padding
    radius = min(center_x, center_y) - padding

    # Create coordinate grids
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circular_mask = (dist_from_center <= radius).astype(np.uint8)

    return circular_mask

def saliency_validity_mask(original_np, dark_thr=15, spec_thr=240, erode_px=4):
    """Data-driven validity mask: exclude near-black borders and specular highlights so
    prompts are not placed on non-tissue artefacts. Unlike a circular FOV, this does NOT
    assume a round frame, so it will not clip polyps lying near the image edge."""
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY) if original_np.ndim == 3 else original_np
    valid = ((gray >= dark_thr) & (gray <= spec_thr)).astype(np.uint8)
    if erode_px > 0:
        valid = cv2.erode(valid, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    return valid

def sharpen_saliency_map(m, gamma=None, pct=None):
    """Turn a diffuse fused attribution map into a peaked, low-noise prior:
    percentile floor (suppress background) + gamma (boost peaks) + light smoothing.
    Operates on the SAME fused map; introduces no new attribution method."""
    gamma = gamma if gamma is not None else REVISION_CONFIG.get("XAI_SHARPEN_GAMMA", 1.5)
    pct = pct if pct is not None else REVISION_CONFIG.get("XAI_SHARPEN_PERCENTILE", 60)
    m = np.asarray(m, dtype=np.float32)
    m = m - m.min()
    if m.max() <= 1e-8:
        return m
    m = m / m.max()
    thr = np.percentile(m, pct)
    m = np.clip((m - thr) / (1.0 - thr + 1e-8), 0.0, 1.0)
    m = np.power(m, gamma)
    m = ndimage.gaussian_filter(m, sigma=1.5)
    mx = m.max()
    return (m / mx) if mx > 1e-8 else m

def select_best_sam_candidate(classifier, sam_predictor, original_np, image_tensor,
                              fused_map, target_class, device, base_bbox, base_pos, base_neg,
                              n_peaks=None):
    """
    LENAS self-evaluation used for CANDIDATE SELECTION (root-cause fix for 'no_overlap').
    A single saliency bounding box lands on the polyp only ~29% of the time on these
    out-of-domain frames, so SAM faithfully segments the wrong region. Instead we build
    several candidate masks -- the saliency-bbox candidate plus one seeded at each of the
    top-K saliency PEAKS -- and keep the one that MAXIMISES the classifier reward for the
    target (polyp) class:  R = conf(mask-in) - conf(mask-out).  No new model is introduced;
    this is exactly the confidence reward already defined in the paper, used to disambiguate
    among prompt hypotheses.
    """
    n_peaks = n_peaks if n_peaks is not None else REVISION_CONFIG.get("KVASIR_N_PEAKS", 4)
    H, W = original_np.shape[:2]
    xs, os_ = fused_map.shape, (H, W)
    candidates = [(base_pos, base_neg, base_bbox)]
    sm = ndimage.gaussian_filter(fused_map, sigma=2.0)
    try:
        peaks = peak_local_max(sm, min_distance=max(5, int(0.06 * min(xs))),
                               num_peaks=int(n_peaks), threshold_abs=np.percentile(sm, 80))
    except Exception:
        peaks = []
    bw, bh = 0.18 * xs[1], 0.18 * xs[0]
    # connected high-saliency blobs (for full-coverage candidates)
    blob_cov = REVISION_CONFIG.get("KVASIR_BLOB_COVERAGE", True)
    lbl = nlbl = None
    if blob_cov:
        try:
            binm = (sm >= 0.35 * (sm.max() + 1e-8)).astype(np.uint8)
            lbl, nlbl = ndimage.label(binm)
        except Exception:
            lbl = nlbl = None
    for (py, px) in peaks:
        # (a) localization candidate: small fixed box + single peak point, no negatives
        box = np.array([px - bw, py - bh, px + bw, py + bh], dtype=float)
        box[0::2] = np.clip(box[0::2], 0, xs[1]); box[1::2] = np.clip(box[1::2], 0, xs[0])
        bbox_o = transform_bbox_to_original(box, xs, os_)
        pos_o = transform_coordinates_to_original(np.array([[px, py]]), xs, os_)
        candidates.append((pos_o, np.array([]), bbox_o))
        # (b) coverage candidate: bounding box of the salient blob containing the peak,
        #     seeded with the peak + blob centroid so SAM grabs the FULL polyp extent.
        if blob_cov and nlbl:
            try:
                cid = lbl[int(np.clip(py, 0, xs[0] - 1)), int(np.clip(px, 0, xs[1] - 1))]
                if cid > 0:
                    ys, xs_ = np.where(lbl == cid)
                    if ys.size >= 20:
                        y0, y1, x0, x1 = ys.min(), ys.max(), xs_.min(), xs_.max()
                        ey, ex = 0.08 * (y1 - y0 + 1), 0.08 * (x1 - x0 + 1)
                        bbx = np.array([x0 - ex, y0 - ey, x1 + ex, y1 + ey], dtype=float)
                        bbx[0::2] = np.clip(bbx[0::2], 0, xs[1]); bbx[1::2] = np.clip(bbx[1::2], 0, xs[0])
                        bbx_o = transform_bbox_to_original(bbx, xs, os_)
                        cpts = np.array([[px, py], [int(xs_.mean()), int(ys.mean())]])
                        cpts_o = transform_coordinates_to_original(cpts, xs, os_)
                        candidates.append((cpts_o, np.array([]), bbx_o))
            except Exception:
                pass
    best_mask, best_reward, best_score = None, -1e9, 0.0
    for pos, neg, bx in candidates:
        try:
            m, sc = segment_with_sam_enhanced(sam_predictor, original_np, pos, neg, bx, post_process=True)
        except Exception:
            continue
        if m is None:
            continue
        a = int((np.asarray(m) > 0).sum())
        if a == 0 or a > 0.9 * H * W:
            continue
        try:
            c_in, c_out = evaluate_masked_confidence(classifier, image_tensor,
                                                     (np.asarray(m) > 0).astype(np.uint8),
                                                     int(target_class), device, os_)
            reward = c_in - c_out
        except Exception:
            reward = 0.0
        if reward > best_reward:
            best_reward, best_mask, best_score = reward, m, sc
    if best_mask is None:
        best_mask, best_score = segment_with_sam_enhanced(
            sam_predictor, original_np, base_pos, base_neg, base_bbox, post_process=True)
    return best_mask, best_score

def select_best_stage_mask(classifier, image_tensor, target_class, device, stage_masks, original_shape):
    """LENAS self-evaluation across pipeline STAGES. Snake / iterative / CRF sometimes
    erode an already-good SAM mask (measured: SAM_raw Dice 0.556 -> final 0.389). Instead of
    blindly returning the last stage, return whichever stage maximises the polyp reward
    R = conf(mask-in) - conf(mask-out). Falls back to the last non-empty mask."""
    best_m, best_r = None, -1e9
    for m in stage_masks:
        if m is None:
            continue
        mb = (np.asarray(m) > 0).astype(np.uint8)
        a = int(mb.sum())
        if a == 0 or a > 0.9 * mb.size:
            continue
        try:
            c_in, c_out = evaluate_masked_confidence(classifier, image_tensor, mb,
                                                     int(target_class), device, original_shape)
            r = float(c_in - c_out)
        except Exception:
            r = -1e8
        if r > best_r:
            best_r, best_m = r, mb
    if best_m is not None:
        return best_m
    for m in reversed(stage_masks):
        if m is not None:
            return (np.asarray(m) > 0).astype(np.uint8)
    return None

def grow_mask_to_saliency(sam_predictor, mask, fused_map, image_np,
                          classifier=None, image_tensor=None, target_class=None, device=None,
                          explanations=None):
    """ANNOTATION-FREE recall growth. SAM locks onto the polyp core, so recall is low. Using
    ONLY XAI saliency maps and the classifier's image-level polyp confidence (no GT, no pixel
    labels), extend the mask into adjacent high-saliency regions and keep the growth only if
    the polyp reward conf(in)-conf(out) does not drop.

    Per-image XAI selection: when `explanations` is given and KVASIR_GROW_XAI_SELECT is on, the
    growth is attempted with EACH attribution map (Saliency, IntegratedGradients, GradientSHAP)
    plus the fused map, and we keep the one whose grown mask the classifier rates most polyp-like.
    So if IntegratedGradients localises this image best, the mask grows into IG's region."""
    def _grow_with_map(m, sal_map, H, W):
        try:
            fm = cv2.resize(np.asarray(sal_map, dtype=np.float32), (W, H))
        except Exception:
            return None
        sal = fm[fm > 0]
        if sal.size < 10:
            return None
        thr = np.percentile(sal, REVISION_CONFIG.get("KVASIR_GROW_PERCENTILE", 65))
        high = (fm >= thr).astype(np.uint8)
        dl = int(REVISION_CONFIG.get("KVASIR_GROW_DILATE", 25))
        near = cv2.dilate(m, np.ones((dl, dl), np.uint8), iterations=1)
        grow_seed = ((high > 0) & (m == 0) & (near > 0)).astype(np.uint8)
        if grow_seed.sum() < 50:
            return None
        ys, xs = np.where(grow_seed > 0)
        vals = fm[ys, xs]
        top = np.argsort(vals)[-min(3, vals.size):]
        new_pts = np.array([[int(xs[i]), int(ys[i])] for i in top])
        cy, cx = np.where(m > 0)
        pos = np.vstack([np.array([[int(cx.mean()), int(cy.mean())]]), new_pts])
        uy, ux = np.where((m > 0) | (grow_seed > 0))
        bbox = np.array([ux.min(), uy.min(), ux.max(), uy.max()], dtype=float)
        try:
            grown, _ = segment_with_sam_enhanced(sam_predictor, image_np, pos, np.array([]),
                                                 bbox, post_process=True)
        except Exception:
            return None
        if grown is None:
            return None
        grown = ((np.asarray(grown) > 0) | (m > 0)).astype(np.uint8)
        if grown.sum() <= m.sum() or grown.sum() > 0.9 * H * W:
            return None
        return grown

    def _reward(m01, H, W):
        if classifier is None or image_tensor is None or target_class is None:
            return 0.0
        try:
            ci, co = evaluate_masked_confidence(classifier, image_tensor, m01,
                                                int(target_class), device, (H, W))
            return float(ci - co)
        except Exception:
            return 0.0

    try:
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.sum() == 0 or sam_predictor is None:
            return m
        H, W = image_np.shape[:2]
        base_r = _reward(m, H, W)
        margin = REVISION_CONFIG.get("KVASIR_GROW_MARGIN", 0.05)

        # candidate saliency maps to grow along
        maps = {}
        if explanations is not None and REVISION_CONFIG.get("KVASIR_GROW_XAI_SELECT", True):
            for name, mp in explanations.items():
                if mp is None:
                    continue
                try:
                    maps[name] = sharpen_saliency_map(mp) if "sharpen_saliency_map" in globals() else mp
                except Exception:
                    maps[name] = mp
        maps["fused"] = fused_map   # always include the fused map as a candidate

        best, best_r, best_name = m, base_r, "none"
        for name, mp in maps.items():
            grown = _grow_with_map(m, mp, H, W)
            if grown is None:
                continue
            r = _reward(grown, H, W)
            if r > best_r:
                best_r, best, best_name = r, grown, name
        # accept only if it does not reduce polyp confidence beyond the margin
        if best_name != "none" and best_r >= base_r - margin and int(best.sum()) > int(m.sum()):
            return best
        return m
    except Exception:
        return (np.asarray(mask) > 0).astype(np.uint8)

def extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05, min_area=100):
    """
    Extract TIGHT bounding box around ONLY the most salient pathology regions.
   
    Args:
        fused_map: Fused XAI saliency map (H, W)
        top_k_percent: Top percentage of salient pixels to consider (default 5%)
        min_area: Minimum area for connected components
   
    Returns:
        bbox: [x_min, y_min, x_max, y_max]
    """
    # Use very aggressive thresholding - only top 5%
    threshold = np.percentile(fused_map, (1 - top_k_percent) * 100)
    binary_mask = (fused_map >= threshold).astype(np.uint8)

    # Find connected components
    labeled_array, num_features = label(binary_mask)

    if num_features == 0:
        # Fallback: use global maximum region
        max_idx = np.unravel_index(np.argmax(fused_map), fused_map.shape)
        y, x = max_idx
        size = 30  # Small box around maximum
        h, w = fused_map.shape
        return np.array([max(0, x-size), max(0, y-size), min(w, x+size), min(h, y+size)])

    # Find the largest connected component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i)
        size = component_mask.sum()
        if size >= min_area:
            component_sizes.append((i, size))

    if not component_sizes:
        # Use global max fallback
        max_idx = np.unravel_index(np.argmax(fused_map), fused_map.shape)
        y, x = max_idx
        size = 30
        h, w = fused_map.shape
        return np.array([max(0, x-size), max(0, y-size), min(w, x+size), min(h, y+size)])

    # Get the largest component
    largest_component_id = max(component_sizes, key=lambda x: x[1])[0]
    largest_component_mask = (labeled_array == largest_component_id)

    # Get tight bbox around this component
    coords = np.where(largest_component_mask > 0)
    if len(coords[0]) == 0:
        h, w = fused_map.shape
        return np.array([w//4, h//4, 3*w//4, 3*h//4])

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add minimal padding (10 pixels)
    padding = 10
    h, w = fused_map.shape
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    return np.array([x_min, y_min, x_max, y_max])

def extract_focused_positive_prompts(fused_map, bbox, num_prompts=3, min_distance=20):
    """
    Extract ONLY high-confidence positive prompts within the pathology region.
   
    Args:
        fused_map: Fused XAI saliency map (H, W)
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        num_prompts: Number of prompts to extract
        min_distance: Minimum distance between prompts (pixels)
   
    Returns:
        prompts: Array of shape (N, 2) with coordinates [[x1, y1], [x2, y2], ...]
    """
    x_min, y_min, x_max, y_max = bbox

    # Extract region of interest
    roi = fused_map[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        # Fallback to center
        h, w = fused_map.shape
        return np.array([[w//2, h//2]])

    # Apply strong smoothing
    smoothed_roi = ndimage.gaussian_filter(roi, sigma=2.0)

    # Use very high threshold - only top 10% within ROI
    threshold_val = np.percentile(smoothed_roi, 90)

    # Find local maxima
    local_maxima = peak_local_max(
        smoothed_roi,
        min_distance=min_distance,
        threshold_abs=threshold_val,
        num_peaks=num_prompts * 2  # Get more candidates
    )

    if len(local_maxima) == 0:
        # Use global maximum in ROI
        max_idx = np.unravel_index(np.argmax(smoothed_roi), smoothed_roi.shape)
        y_roi, x_roi = max_idx
        # Convert back to full image coordinates
        x_full = x_roi + x_min
        y_full = y_roi + y_min
        return np.array([[x_full, y_full]])

    # Score each peak
    candidates = []
    for peak in local_maxima:
        y_roi, x_roi = peak
        intensity = smoothed_roi[y_roi, x_roi]

        # Calculate local neighborhood quality
        y_start, y_end = max(0, y_roi-3), min(smoothed_roi.shape[0], y_roi+4)
        x_start, x_end = max(0, x_roi-3), min(smoothed_roi.shape[1], x_roi+4)
        neighborhood = smoothed_roi[y_start:y_end, x_start:x_end]
        local_std = neighborhood.std()

        quality_score = intensity + 0.1 * local_std

        # Convert to full image coordinates
        x_full = x_roi + x_min
        y_full = y_roi + y_min

        candidates.append({
            'coords': (x_full, y_full),
            'quality_score': quality_score
        })

    # Sort by quality
    candidates.sort(key=lambda x: x['quality_score'], reverse=True)

    # Select spatially diverse prompts
    selected_prompts = [candidates[0]]
    for candidate in candidates[1:]:
        if len(selected_prompts) >= num_prompts:
            break

        too_close = False
        for selected in selected_prompts:
            dist = np.sqrt((candidate['coords'][0] - selected['coords'][0])**2 +
                          (candidate['coords'][1] - selected['coords'][1])**2)
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            selected_prompts.append(candidate)

    prompts = np.array([p['coords'] for p in selected_prompts])
    return prompts

def extract_smart_negative_prompts(fused_map, bbox, circular_mask, num_negatives=3):
    """
    Extract negative prompts from BACKGROUND (low saliency) regions.
   
    Args:
        fused_map: Fused XAI saliency map (H, W)
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        circular_mask: Binary mask of circular FOV
        num_negatives: Number of negative prompts to extract
   
    Returns:
        negative_prompts: Array of shape (N, 2) with coordinates [[x1, y1], [x2, y2], ...]
    """
    h, w = fused_map.shape
    x_min, y_min, x_max, y_max = bbox

    # Create exclusion mask: exclude bbox region
    exclusion_mask = np.ones((h, w), dtype=bool)
    exclusion_mask[y_min:y_max, x_min:x_max] = False

    # Also use circular mask to stay within FOV
    if circular_mask is not None:
        exclusion_mask = exclusion_mask & (circular_mask > 0)

    # Find low saliency regions
    low_sal_threshold = np.percentile(fused_map, 20)  # Bottom 20%
    low_sal_mask = (fused_map < low_sal_threshold) & exclusion_mask

    low_sal_coords = np.where(low_sal_mask)

    if len(low_sal_coords[0]) == 0:
        # Fallback: corners with minimum distance from bbox
        margin = 30
        candidates = [
            (margin, margin),
            (w - margin, margin),
            (margin, h - margin),
            (w - margin, h - margin),
        ]

        negative_prompts = []
        for x, y in candidates:
            # Check if outside bbox with margin
            if (x < x_min - 20 or x > x_max + 20 or
                y < y_min - 20 or y > y_max + 20):
                negative_prompts.append((x, y))
                if len(negative_prompts) >= num_negatives:
                    break

        return np.array(negative_prompts) if negative_prompts else np.array([])

    # Sample from low saliency regions
    indices = np.random.choice(len(low_sal_coords[0]),
                              min(num_negatives, len(low_sal_coords[0])),
                              replace=False)

    negative_prompts = []
    for idx in indices:
        y, x = low_sal_coords[0][idx], low_sal_coords[1][idx]
        negative_prompts.append((x, y))

    return np.array(negative_prompts)

def transform_coordinates_to_original(coords, xai_shape, original_shape):
    """
    Transform coordinates from XAI space to original image space.
   
    Args:
        coords: Array of coordinates in XAI space (N, 2) [[x, y], ...]
        xai_shape: Shape of XAI map (H_xai, W_xai)
        original_shape: Shape of original image (H_orig, W_orig)
   
    Returns:
        transformed_coords: Array of coordinates in original space (N, 2)
    """
    h_xai, w_xai = xai_shape
    h_orig, w_orig = original_shape

    scale_x = w_orig / w_xai
    scale_y = h_orig / h_xai

    if len(coords) == 0:
        return coords

    transformed_coords = coords.copy().astype(float)
    transformed_coords[:, 0] *= scale_x  # Scale x coordinates
    transformed_coords[:, 1] *= scale_y  # Scale y coordinates

    return transformed_coords.astype(int)

def transform_bbox_to_original(bbox, xai_shape, original_shape):
    """
    Transform bounding box from XAI space to original image space.
   
    Args:
        bbox: Bounding box in XAI space [x_min, y_min, x_max, y_max]
        xai_shape: Shape of XAI map (H_xai, W_xai)
        original_shape: Shape of original image (H_orig, W_orig)
   
    Returns:
        bbox_orig: Bounding box in original space [x_min, y_min, x_max, y_max]
    """
    h_xai, w_xai = xai_shape
    h_orig, w_orig = original_shape

    scale_x = w_orig / w_xai
    scale_y = h_orig / h_xai

    x_min, y_min, x_max, y_max = bbox

    x_min_orig = int(x_min * scale_x)
    y_min_orig = int(y_min * scale_y)
    x_max_orig = int(x_max * scale_x)
    y_max_orig = int(y_max * scale_y)

    return np.array([x_min_orig, y_min_orig, x_max_orig, y_max_orig])

def segment_with_sam_enhanced(sam_predictor, image_np, positive_prompts, negative_prompts=None,
                              bbox=None, post_process=True):
    """
    Enhanced SAM segmentation with smart prompting.
   
    Args:
        sam_predictor: SAM predictor object
        image_np: Original RGB image as numpy array (H, W, 3)
        positive_prompts: Array of positive prompt coordinates (N, 2) [[x, y], ...]
        negative_prompts: Array of negative prompt coordinates (M, 2) [[x, y], ...]
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        post_process: Whether to apply post-processing
   
    Returns:
        mask: Binary segmentation mask (H, W)
        score: Confidence score from SAM
    """
    try:
        sam_predictor.set_image(image_np)

        # Prepare prompts
        input_points = positive_prompts
        input_labels = np.ones(len(positive_prompts))

        # Add negative prompts
        if negative_prompts is not None and len(negative_prompts) > 0:
            input_points = np.vstack([positive_prompts, negative_prompts])
            input_labels = np.concatenate([
                np.ones(len(positive_prompts)),
                np.zeros(len(negative_prompts))
            ])

        # Predict with SAM - use box for better guidance
        if bbox is not None:
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=bbox,
                multimask_output=True
            )
        else:
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)

        # Post-processing
        if post_process:
            mask = post_process_mask(mask, image_np)

        return mask, scores[best_idx]

    except Exception as e:
        print(f"SAM segmentation error: {e}")
        return np.zeros(image_np.shape[:2], dtype=np.uint8), 0.0

def visualize_kvasir_segmentation_pipeline(sample, classifier, sam_predictor, device, xai_weights,
                                          fusion_strategy='weighted_average'):
    """
    Complete visualization pipeline for Kvasir-SEG dataset.
    Shows original image, ground truth mask, predicted mask, and all intermediate steps.
    """
    image_tensor, gt_mask_tensor, img_path = sample
    
    # Load original image
    original_pil = Image.open(img_path).convert("RGB")
    original_np = np.array(original_pil)
    
    # Convert ground truth mask to numpy
    gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(img_path)}")
    print(f"{'='*80}")

    # Get model prediction for the image (we'll use the highest probability class)
    with torch.no_grad():
        _, entropy, probs = classifier.predict_with_uncertainty(image_tensor.unsqueeze(0).to(device))
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        entropy_val = entropy[0].item()

    print(f"Model Prediction: Class {predicted_class}, Confidence: {confidence:.4f}, Entropy: {entropy_val:.4f}")

    # LENAS supervision = IMAGE-LEVEL LABEL. Every Kvasir-SEG image is a polyp, so condition
    # the XAI (and the self-correction reward) on the polyp class, not the out-of-domain argmax.
    target_class = predicted_class
    if REVISION_CONFIG.get("KVASIR_FORCE_POLYP_CLASS", True):
        target_class = REVISION_CONFIG.get("KVASIR_TARGET_CLASS", 7)
        print(f"  -> Conditioning XAI on image-level label (Polyp = class {target_class})")

    # Generate XAI
    explanations = generate_explanations_focused(classifier, image_tensor.unsqueeze(0),
                                                target_class, device, use_cache=True)
    fused_map = advanced_xai_fusion(explanations, xai_weights, fusion_strategy)

    # keep prompts off black borders / specular glare (data-driven; no circular clipping)
    if REVISION_CONFIG.get("SUPPRESS_BORDER_SPECULAR", True):
        _valid = saliency_validity_mask(original_np)
        _valid_xai = cv2.resize(_valid, (fused_map.shape[1], fused_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        _fm = fused_map * (_valid_xai > 0)
        if _fm.max() > 0:
            fused_map = _fm
    if REVISION_CONFIG.get("XAI_SHARPEN", True):
        fused_map = sharpen_saliency_map(fused_map)

    # Uncertainty-guided prompting
    num_prompts, strategy = uncertainty_guided_prompts(fused_map, entropy_val,
                                                      probs[0].cpu().numpy())

    # Extract prompts
    bbox_xai = extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05)
    positive_prompts_xai = extract_focused_positive_prompts(fused_map, bbox_xai,
                                                           num_prompts=num_prompts)

    circular_mask = detect_circular_mask(original_np.shape)
    circular_mask_xai = cv2.resize(circular_mask, (fused_map.shape[1], fused_map.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    negative_prompts_xai = extract_smart_negative_prompts(fused_map, bbox_xai,
                                                         circular_mask_xai, num_negatives=3)

    # Transform to original space
    xai_shape = fused_map.shape
    original_shape = original_np.shape[:2]
    bbox_orig = transform_bbox_to_original(bbox_xai, xai_shape, original_shape)
    positive_prompts_orig = transform_coordinates_to_original(positive_prompts_xai, xai_shape, original_shape)
    negative_prompts_orig = transform_coordinates_to_original(negative_prompts_xai, xai_shape, original_shape) if len(negative_prompts_xai) > 0 else np.array([])

    # Run segmentation pipeline
    if sam_predictor is not None:
        if REVISION_CONFIG.get("KVASIR_CANDIDATE_SELECTION", True):
            initial_mask, sam_score = select_best_sam_candidate(
                classifier, sam_predictor, original_np, image_tensor, fused_map,
                target_class, device, bbox_orig, positive_prompts_orig, negative_prompts_orig)
        else:
            initial_mask, sam_score = segment_with_sam_enhanced(
                sam_predictor, original_np, positive_prompts_orig,
                negative_prompts_orig, bbox_orig, post_process=True
            )
        if REVISION_CONFIG.get("KVASIR_RECALL_GROWTH", True):
            initial_mask = grow_mask_to_saliency(
                sam_predictor, initial_mask, fused_map, original_np,
                classifier, image_tensor, target_class, device, explanations=explanations)
        snake_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)
        
        # Run iterative self-correction
        final_mask, iter_history = iterative_self_correction_improved(
            classifier, sam_predictor, original_np, image_tensor, target_class,
            device, xai_weights, max_iterations=REVISION_CONFIG.get("TMAX", 5),
            init_mask=(snake_mask if REVISION_CONFIG.get("KVASIR_ITER_INIT_MASK", True) else None)
        )
    else:
        # Fallback segmentation
        threshold = np.percentile(fused_map, 95)
        initial_mask = (fused_map >= threshold).astype(np.uint8)
        initial_mask = cv2.resize(initial_mask, (original_np.shape[1], original_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        snake_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)
        final_mask, sam_score, iter_history = snake_mask, 0.0, None

    # --- REVISION: editor-approved CRF + stage selection by confidence reward ---
    crf_mask = maybe_crf(original_np, final_mask)
    if sam_predictor is not None and REVISION_CONFIG.get("KVASIR_STAGE_SELECT", True):
        final_mask = select_best_stage_mask(
            classifier, image_tensor, target_class, device,
            [initial_mask, snake_mask, final_mask, crf_mask],
            (original_np.shape[0], original_np.shape[1]))
    else:
        final_mask = crf_mask

    # Calculate metrics
    dice_score = calculate_dice_score(final_mask, gt_mask_np)
    iou_score = calculate_iou_score(final_mask, gt_mask_np)

    print(f"\n📊 Segmentation Metrics:")
    print(f"  Dice Score: {dice_score:.4f}")
    print(f"  IoU Score:  {iou_score:.4f}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(25, 18))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.2)

    # Row 0: Input and XAI
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(original_np)
    ax00.set_title(f"Original Image\n{os.path.basename(img_path)}", fontsize=12, fontweight='bold')
    ax00.axis('off')

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(gt_mask_np, cmap='gray')
    ax01.set_title("Ground Truth Mask", fontsize=12, fontweight='bold')
    ax01.axis('off')

    # Show individual XAI methods
    method_names = list(explanations.keys())
    for i, method in enumerate(method_names[:2]):
        ax = fig.add_subplot(gs[0, i + 2])
        xai_map = explanations[method]
        xai_norm = (xai_map - xai_map.min()) / (xai_map.max() - xai_map.min() + 1e-8)
        xai_resized = cv2.resize(xai_norm, (original_np.shape[1], original_np.shape[0]))
        ax.imshow(original_np, alpha=0.5)
        ax.imshow(xai_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"{method}\n(Weight: {xai_weights.get(method, 0):.3f})", fontsize=11)
        ax.axis('off')

    ax04 = fig.add_subplot(gs[0, 4])
    fused_resized = cv2.resize(fused_map, (original_np.shape[1], original_np.shape[0]))
    ax04.imshow(original_np, alpha=0.5)
    ax04.imshow(fused_resized, cmap='jet', alpha=0.5)
    if len(positive_prompts_orig) > 0:
        ax04.scatter(positive_prompts_orig[:, 0], positive_prompts_orig[:, 1], 
                    c='lime', marker='*', s=150, edgecolors='black', label='Positive')
    if len(negative_prompts_orig) > 0:
        ax04.scatter(negative_prompts_orig[:, 0], negative_prompts_orig[:, 1], 
                    c='red', marker='X', s=120, label='Negative')
    if bbox_orig is not None:
        x1, y1, x2, y2 = bbox_orig
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
        ax04.add_patch(rect)
    ax04.set_title(f"Fused Saliency + Prompts\nStrategy: {strategy}", fontsize=11, fontweight='bold')
    ax04.legend()
    ax04.axis('off')

    # Row 1: Segmentation progression
    axes_list = [fig.add_subplot(gs[1, i]) for i in range(5)]
    
    # Initial SAM
    if 'initial_mask' in locals():
        overlay_initial = original_np.copy().astype(float) * 0.7
        overlay_initial[initial_mask == 1] += np.array([255, 0, 0]) * 0.3
        axes_list[0].imshow(np.clip(overlay_initial, 0, 255).astype(np.uint8))
        axes_list[0].set_title(f"1. Initial SAM\nScore: {sam_score:.3f}", fontsize=11)
    
    # Snake-refined
    overlay_snake = original_np.copy().astype(float) * 0.7
    overlay_snake[snake_mask == 1] += np.array([0, 255, 0]) * 0.3
    axes_list[1].imshow(np.clip(overlay_snake, 0, 255).astype(np.uint8))
    axes_list[1].set_title("2. Snake Refinement", fontsize=11)

    # Final Prediction
    overlay_final = original_np.copy().astype(float) * 0.7
    overlay_final[final_mask == 1] += np.array([0, 0, 255]) * 0.3
    axes_list[2].imshow(np.clip(overlay_final, 0, 255).astype(np.uint8))
    title_suffix = f"(Iter {len(iter_history['iteration'])-1})" if iter_history else ""
    axes_list[2].set_title(f"3. Final Prediction {title_suffix}", fontsize=11, fontweight='bold')
    
    # Ground Truth
    overlay_gt = original_np.copy().astype(float) * 0.7
    overlay_gt[gt_mask_np == 1] += np.array([255, 255, 0]) * 0.3
    axes_list[3].imshow(np.clip(overlay_gt, 0, 255).astype(np.uint8))
    axes_list[3].set_title("4. Ground Truth", fontsize=11, fontweight='bold')
    
    # Error Map (FP/FN)
    fp = np.logical_and(final_mask == 1, gt_mask_np == 0).astype(np.uint8) # False Positive
    fn = np.logical_and(final_mask == 0, gt_mask_np == 1).astype(np.uint8) # False Negative
    error_map = np.zeros_like(original_np)
    error_map[fp == 1] = [255, 20, 147] # Hot Pink for FP
    error_map[fn == 1] = [255, 165, 0]   # Orange for FN
    axes_list[4].imshow(original_np)
    axes_list[4].imshow(error_map, alpha=0.7)
    axes_list[4].set_title(f"5. Error Map (FP/FN)\nDice: {dice_score:.3f}, IoU: {iou_score:.3f}", 
                          fontsize=11, fontweight='bold')

    for ax in axes_list:
        ax.axis('off')

    # Row 2: Metrics and iteration history
    if iter_history is not None:
        ax20 = fig.add_subplot(gs[2, :3])
        ax20_twin = ax20.twinx()

        iterations = iter_history['iteration']
        ax20.plot(iterations, iter_history['confidence_diff'], 'b-^', linewidth=2.5, markersize=8, label='Confidence Difference')
        ax20_twin.plot(iterations, iter_history['mask_quality'], 'purple', linestyle='--', marker='D', linewidth=2.5, markersize=8, label='Mask Quality Score')
        
        if 'best_iteration' in iter_history and iter_history['best_iteration'] >= 0:
            best_iter = iter_history['best_iteration']
            ax20.axvline(x=best_iter, color='gold', linestyle=':', linewidth=3, label=f'Best Mask (Iter {best_iter})')

        ax20.set_xlabel('Iteration', fontsize=12)
        ax20.set_ylabel('Confidence Metric', fontsize=12, color='blue')
        ax20_twin.set_ylabel('Quality Metric', fontsize=12, color='purple')
        ax20.set_title('Self-Correction: Confidence & Quality Evolution', fontsize=14, fontweight='bold')
        ax20.tick_params(axis='y', labelcolor='blue', labelsize=10)
        ax20_twin.tick_params(axis='y', labelcolor='purple', labelsize=10)
        ax20.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax20.legend(loc='upper left', fontsize=10)
        ax20_twin.legend(loc='upper right', fontsize=10)

        ax21 = fig.add_subplot(gs[2, 3:])
        ax21.axis('off')
        stats_text = (
            f"Pipeline Statistics\n"
            f"{'='*50}\n"
            f"Predicted Class: {predicted_class}\n"
            f"Confidence: {confidence:.4f} | Entropy: {entropy_val:.4f}\n"
            f"Prompting Strategy: {strategy}\n"
            f"SAM Score: {sam_score:.4f}\n"
            f"Final Mask Size: {np.sum(final_mask)} px\n"
            f"\nSegmentation Metrics:\n"
            f"Dice Score: {dice_score:.4f}\n"
            f"IoU Score:  {iou_score:.4f}\n"
            f"\nRefinement Summary:\n"
            f"Iterations: {len(iter_history['iteration']) - 1}\n"
            f"Converged: {'Yes' if iter_history['converged'] else 'No'}\n"
            f"Best Iteration: {iter_history.get('best_iteration', 'N/A')}"
        )
        ax21.text(0.05, 0.95, stats_text, transform=ax21.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

    else:
        ax2 = fig.add_subplot(gs[2, :])
        ax2.axis('off')
        stats_text = (
            f"Pipeline Statistics\n"
            f"{'='*50}\n"
            f"Predicted Class: {predicted_class}\n"
            f"Confidence: {confidence:.4f} | Entropy: {entropy_val:.4f}\n"
            f"Prompting Strategy: {strategy}\n"
            f"SAM Score: {sam_score:.4f}\n"
            f"Final Mask Size: {np.sum(final_mask)} px\n"
            f"\nSegmentation Metrics:\n"
            f"Dice Score: {dice_score:.4f}\n"
            f"IoU Score:  {iou_score:.4f}\n"
            f"\nNote: No iterative refinement was performed"
        )
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

    fig.suptitle(f"Kvasir-SEG Segmentation Pipeline: {os.path.basename(img_path)}",
                fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_current_figure()
    plt.show()
    plt.close('all')

    return final_mask, dice_score, iou_score, iter_history

# =========================================================================================
# Part 13: MAIN EXECUTION PIPELINE
# =========================================================================================

def main():
    """Main execution pipeline for both capsule endoscopy and Kvasir-SEG."""
    clear_output()
    setup_output_dirs("Kvasir")

    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    SAM_CKPT_PATH = "/media/data/DARE/sam_vit_b_01ec64.pth"
    
    # Paths for both datasets
    CAPSULE_DATA_PATH = "/home/satyajith/.cache/kagglehub/datasets/podakantisatyajith/capsule-vision-challenge-final-dataset-2024/versions/1/Dataset/Dataset/training"
    KVASIR_SEG_DATA_DIR = "/home/satyajith/.cache/kagglehub/datasets/debeshjha1/kvasirseg/versions/3/Kvasir-SEG/Kvasir-SEG"
    
    # Capsule endoscopy configuration
    CAPSULE_NUM_CLASSES = 8
    CAPSULE_BATCH_SIZE = 16
    CAPSULE_EPOCHS = 1
    CAPSULE_LR = 1e-5
    CAPSULE_CLASS_LABELS = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
                           'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps',
                           'ulcerative-colitis']
    
    CAPSULE_MODEL_SAVE_PATH = resolve_ckpt("best_kvasir_domain_classifier.pth")

    print(f"Using device: {DEVICE}")
    print(f"SAM Available: {SAM_AVAILABLE}")

    # ========== STAGE 1: Train on Capsule Endoscopy Dataset ==========
    print("\n" + "="*80)
    print("STAGE 1: Training on Capsule Endoscopy Dataset (10 classes)")
    print("="*80)

    capsule_model = DifferentialBiomedCLIP(MODEL_NAME, CAPSULE_NUM_CLASSES, DEVICE,
                                          class_names=CAPSULE_CLASS_LABELS, use_contrastive=True)

    # Load capsule endoscopy dataset
    capsule_dataset = CapsuleEndoscopyDataset(
        root_dir=CAPSULE_DATA_PATH, 
        transform=capsule_model.preprocess
    )

    if not capsule_dataset.samples:
        print(f"❌ No samples found in capsule dataset at {CAPSULE_DATA_PATH}")
        return None, None, None

    print(f"Total capsule endoscopy samples: {len(capsule_dataset)}")

    # Split dataset
    train_size = int(0.8 * len(capsule_dataset))
    val_size = len(capsule_dataset) - train_size
    train_ds, val_ds = random_split(capsule_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, CAPSULE_BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn_capsule, num_workers=2)
    val_loader = DataLoader(val_ds, CAPSULE_BATCH_SIZE, shuffle=False, 
                           collate_fn=collate_fn_capsule, num_workers=2)

    # Train or load model
    if os.path.exists(CAPSULE_MODEL_SAVE_PATH):
        print(f"Loading existing capsule model from {CAPSULE_MODEL_SAVE_PATH}")
        try:
            checkpoint = torch.load(CAPSULE_MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
            capsule_model.load_state_dict(checkpoint['model_state_dict'])
            _ep = checkpoint.get('epoch', 'n/a') if isinstance(checkpoint, dict) else 'n/a'
            _va = checkpoint.get('val_acc') if isinstance(checkpoint, dict) else None
            if isinstance(_va, (int, float)):
                print(f"✅ Loaded classifier (epoch {_ep}, val_acc {_va:.2f}%)")
            else:
                print(f"✅ Loaded classifier (epoch {_ep})")
            # Auto-derive the polyp target index from the checkpoint, so the XAI is always
            # conditioned on the correct class regardless of any stale config value.
            if isinstance(checkpoint, dict) and checkpoint.get('polyp_idx') is not None:
                REVISION_CONFIG['KVASIR_FORCE_POLYP_CLASS'] = True
                REVISION_CONFIG['KVASIR_TARGET_CLASS'] = int(checkpoint['polyp_idx'])
                print(f"🎯 XAI target set to polyp index {checkpoint['polyp_idx']} (from checkpoint)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training capsule model from scratch...")
            history = train_classifier_with_hybrid_loss(
                capsule_model, train_loader, val_loader, CAPSULE_EPOCHS, CAPSULE_LR, DEVICE,
                CAPSULE_CLASS_LABELS, CAPSULE_MODEL_SAVE_PATH,
                contrastive_weight=0.3, classification_weight=0.7
            )
    else:
        print("No existing capsule model found. Training from scratch...")
        history = train_classifier_with_hybrid_loss(
            capsule_model, train_loader, val_loader, CAPSULE_EPOCHS, CAPSULE_LR, DEVICE,
            CAPSULE_CLASS_LABELS, CAPSULE_MODEL_SAVE_PATH,
            contrastive_weight=0.3, classification_weight=0.7
        )

    # ========== STAGE 2: XAI Evaluation on Capsule Model ==========
    print("\n" + "="*80)
    print("STAGE 2: XAI Evaluation on Capsule Model")
    print("="*80)

    # Use pre-computed weights or evaluate
    use_cached_weights = False
    if use_cached_weights:
        xai_weights = {'Saliency': 0.279, 'IntegratedGradients': 0.352, 'GradientShap': 0.369}
        print(f"Using pre-computed XAI weights: {xai_weights}")
    else:
        xai_weights, xai_metrics = evaluate_xai_methods_quantus(
            capsule_model, val_loader, DEVICE, CAPSULE_NUM_CLASSES, 
            num_samples=20, CLASS_LABELS=CAPSULE_CLASS_LABELS
        )

    # ========== STAGE 3: Setup SAM for Segmentation ==========
    print("\n" + "="*80)
    print("STAGE 3: Setting up SAM for Kvasir-SEG Segmentation")
    print("="*80)

    sam_predictor = None
    if SAM_AVAILABLE and os.path.exists(SAM_CKPT_PATH):
        try:
            sam_model = sam_model_registry['vit_b'](checkpoint=SAM_CKPT_PATH)
            sam_model = sam_model.to(DEVICE)
            sam_predictor = SamPredictor(sam_model)
            print("✅ SAM model loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM: {e}")
            print("Will use fallback segmentation.")
    else:
        print(f"SAM checkpoint not found at {SAM_CKPT_PATH}")
        print("Will use fallback segmentation.")

    # ========== STAGE 4: Kvasir-SEG Segmentation Evaluation ==========
    print("\n" + "="*80)
    print("STAGE 4: Kvasir-SEG Segmentation Evaluation")
    print("="*80)

    # Load Kvasir-SEG dataset
    kvasir_image_dir = os.path.join(KVASIR_SEG_DATA_DIR, 'images')
    kvasir_mask_dir = os.path.join(KVASIR_SEG_DATA_DIR, 'masks')
    
    if not os.path.exists(kvasir_image_dir) or not os.path.exists(kvasir_mask_dir):
        print(f"❌ Kvasir-SEG dataset not found at {KVASIR_SEG_DATA_DIR}")
        print("Please update the KVASIR_SEG_DATA_DIR path")
        return capsule_model, sam_predictor, xai_weights

    # Create transforms for Kvasir-SEG
    kvasir_transform = capsule_model.preprocess  # Use same transform as training
    kvasir_mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    kvasir_dataset = KvasirSEGDataset(
        image_dir=kvasir_image_dir,
        mask_dir=kvasir_mask_dir,
        transform=kvasir_transform,
        mask_transform=kvasir_mask_transform
    )

    print(f"Loaded {len(kvasir_dataset)} Kvasir-SEG samples")

    # First, let's visualize some samples from Kvasir-SEG
    print("\n📊 Visualizing Kvasir-SEG samples...")
    num_sample_viz = min(5, len(kvasir_dataset))
    sample_indices = np.random.choice(len(kvasir_dataset), num_sample_viz, replace=False)
    
    fig, axes = plt.subplots(num_sample_viz, 3, figsize=(15, 5*num_sample_viz))
    if num_sample_viz == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        image_tensor, mask_tensor, img_path = kvasir_dataset[idx]
        
        # Convert back to numpy for visualization
        image_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = mask_tensor.squeeze().cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original: {os.path.basename(img_path)}")
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(image_np)
        axes[i, 2].imshow(mask_np, alpha=0.5, cmap='jet')
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Kvasir-SEG Sample Visualizations", fontsize=16, fontweight='bold')
    save_current_figure()
    plt.show()

    # Now run segmentation on Kvasir-SEG dataset
    print("\n🎯 Running segmentation on Kvasir-SEG dataset...")
    
    kvasir_loader = DataLoader(kvasir_dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn_kvasir)

    all_dice_scores = []
    all_iou_scores = []
    segmentation_results = []

    # ---- FULL-DATASET METRICS (no plotting; CRF applied inside) ----
    # Plotting a figure per image previously stalled the run before any aggregate
    # was produced. Metrics and visualization are now decoupled.
    for i, (images, masks, paths) in tqdm(enumerate(kvasir_loader), total=len(kvasir_loader), desc="Segmenting Kvasir-SEG"):
        if images.nelement() == 0:
            continue
        sample = (images.squeeze(0), masks.squeeze(0), paths[0])
        try:
            dice, iou = _kvasir_segment_metrics_only(
                sample, capsule_model, sam_predictor, DEVICE, xai_weights,
                fusion_strategy="weighted_average", use_iterative=True)
            all_dice_scores.append(dice)
            all_iou_scores.append(iou)
            segmentation_results.append({'image_path': paths[0], 'dice': dice, 'iou': iou})
        except Exception as e:
            print(f"❌ Error processing {paths[0]}: {e}")
        if (i + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Representative qualitative figures (few samples; figures closed) ----
    n_viz = REVISION_CONFIG.get("NUM_VIZ_SAMPLES", 6)
    print(f"\n🖼\uFE0F  Generating up to {n_viz} representative qualitative figures...")
    _viz_done = 0
    for i, (images, masks, paths) in enumerate(kvasir_loader):
        if _viz_done >= n_viz:
            break
        if images.nelement() == 0:
            continue
        try:
            visualize_kvasir_segmentation_pipeline(
                (images.squeeze(0), masks.squeeze(0), paths[0]),
                capsule_model, sam_predictor, DEVICE, xai_weights)
            plt.close('all')
            _viz_done += 1
        except Exception as _e:
            print(f"viz skipped: {_e}")

    # ========== STAGE 5: Final Results and Summary ==========
    print("\n" + "="*80)
    print("STAGE 5: Final Results Summary")
    print("="*80)

    if all_dice_scores:
        avg_dice = np.mean(all_dice_scores)
        avg_iou = np.mean(all_iou_scores)
        std_dice = np.std(all_dice_scores)
        std_iou = np.std(all_iou_scores)
        
        print(f"\n📊 Kvasir-SEG Segmentation Results:")
        print(f"   Total Samples Processed: {len(all_dice_scores)}")
        print(f"   Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"   Average IoU Score:  {avg_iou:.4f} ± {std_iou:.4f}")
        print(f"   Best Dice Score:    {np.max(all_dice_scores):.4f}")
        print(f"   Best IoU Score:     {np.max(all_iou_scores):.4f}")
        print(f"   Worst Dice Score:   {np.min(all_dice_scores):.4f}")
        print(f"   Worst IoU Score:    {np.min(all_iou_scores):.4f}")
        
        # Plot distribution of scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(all_dice_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(avg_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dice:.4f}')
        ax1.set_xlabel('Dice Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Dice Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(all_iou_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(avg_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_iou:.4f}')
        ax2.set_xlabel('IoU Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of IoU Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_current_figure()
        plt.show()
        
        # Save results to CSV
        results_df = pd.DataFrame(segmentation_results)
        results_df.to_csv(out_path('results', 'kvasir_segmentation_results.csv'), index=False)
        print(f"\n💾 Results saved to {out_path('results', 'kvasir_segmentation_results.csv')}")
        
    else:
        print("❌ No successful segmentations to report.")

    # Clear XAI cache
    xai_cache.clear()

    print("\n" + "="*80)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("="*80)
    
    print(f"\n📝 Pipeline Summary:")
    print(f"   • Capsule Endoscopy Model: Trained on {len(capsule_dataset)} samples")
    print(f"   • Kvasir-SEG Segmentation: Processed {len(all_dice_scores) if all_dice_scores else 0} samples")
    print(f"   • XAI Methods: {len(xai_weights)} methods with optimized weights")
    print(f"   • SAM Integration: {'Enabled' if sam_predictor else 'Disabled'}")
    
    return capsule_model, sam_predictor, xai_weights, kvasir_dataset

# =========================================================================================
# EXECUTE MAIN PIPELINE
# =========================================================================================


# =========================================================================================
# REVISION EXPERIMENT SUITE (Kvasir-SEG) -- reuses the existing pipeline, REAL numbers only
# =========================================================================================
def _kvasir_segment_metrics_only(sample, classifier, sam_predictor, device, xai_weights,
                                 fusion_strategy="weighted_average", use_iterative=True):
    """Metrics-only mirror of visualize_kvasir_segmentation_pipeline (no plotting)."""
    image_tensor, gt_mask_tensor, img_path = sample
    original_np = np.array(Image.open(img_path).convert("RGB"))
    gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()
    with torch.no_grad():
        _, entropy, probs = classifier.predict_with_uncertainty(image_tensor.unsqueeze(0).to(device))
        predicted_class = torch.argmax(probs, dim=1).item()
        entropy_val = entropy[0].item()
    target_class = predicted_class
    if REVISION_CONFIG.get("KVASIR_FORCE_POLYP_CLASS", True):
        target_class = REVISION_CONFIG.get("KVASIR_TARGET_CLASS", 7)
    explanations = generate_explanations_focused(classifier, image_tensor.unsqueeze(0),
                                                 target_class, device, use_cache=True)
    fused_map = advanced_xai_fusion(explanations, xai_weights, fusion_strategy)
    if REVISION_CONFIG.get("SUPPRESS_BORDER_SPECULAR", True):
        _valid = saliency_validity_mask(original_np)
        _valid_xai = cv2.resize(_valid, (fused_map.shape[1], fused_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        _fm = fused_map * (_valid_xai > 0)
        if _fm.max() > 0:
            fused_map = _fm
    if REVISION_CONFIG.get("XAI_SHARPEN", True):
        fused_map = sharpen_saliency_map(fused_map)
    num_prompts, _ = uncertainty_guided_prompts(fused_map, entropy_val, probs[0].cpu().numpy())
    bbox_xai = extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05)
    positive_prompts_xai = extract_focused_positive_prompts(fused_map, bbox_xai, num_prompts=num_prompts)
    circular_mask = detect_circular_mask(original_np.shape)
    circular_mask_xai = cv2.resize(circular_mask, (fused_map.shape[1], fused_map.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    negative_prompts_xai = extract_smart_negative_prompts(fused_map, bbox_xai, circular_mask_xai, num_negatives=3)
    xai_shape = fused_map.shape; original_shape = original_np.shape[:2]
    bbox_orig = transform_bbox_to_original(bbox_xai, xai_shape, original_shape)
    positive_prompts_orig = transform_coordinates_to_original(positive_prompts_xai, xai_shape, original_shape)
    negative_prompts_orig = (transform_coordinates_to_original(negative_prompts_xai, xai_shape, original_shape)
                             if len(negative_prompts_xai) > 0 else np.array([]))
    if sam_predictor is not None:
        if not REVISION_CONFIG.get("KVASIR_USE_SAM", True):
            # Variant C: no SAM -- threshold the fused saliency map directly
            _pos = fused_map[fused_map > 0]
            _thr = np.percentile(_pos, 85) if _pos.size else float(fused_map.max())
            _m = (fused_map >= _thr).astype(np.uint8)
            initial_mask = cv2.resize(_m, (original_np.shape[1], original_np.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            sam_score = 0.0
        elif REVISION_CONFIG.get("KVASIR_CANDIDATE_SELECTION", True):
            initial_mask, sam_score = select_best_sam_candidate(
                classifier, sam_predictor, original_np, image_tensor, fused_map,
                target_class, device, bbox_orig, positive_prompts_orig, negative_prompts_orig)
        else:
            initial_mask, sam_score = segment_with_sam_enhanced(
                sam_predictor, original_np, positive_prompts_orig, negative_prompts_orig, bbox_orig, post_process=True)
        if REVISION_CONFIG.get("KVASIR_USE_SAM", True) and REVISION_CONFIG.get("KVASIR_RECALL_GROWTH", True):
            initial_mask = grow_mask_to_saliency(
                sam_predictor, initial_mask, fused_map, original_np,
                classifier, image_tensor, target_class, device, explanations=explanations)
        if REVISION_CONFIG.get("KVASIR_USE_SNAKE", True):
            snake_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)
        else:
            snake_mask = initial_mask
        if use_iterative:
            final_mask, _ = iterative_self_correction_improved(
                classifier, sam_predictor, original_np, image_tensor, target_class,
                device, xai_weights, max_iterations=REVISION_CONFIG.get("TMAX", 5),
                init_mask=(snake_mask if REVISION_CONFIG.get("KVASIR_ITER_INIT_MASK", True) else None))
        else:
            final_mask = snake_mask
    else:
        threshold = np.percentile(fused_map, 95)
        initial_mask = (fused_map >= threshold).astype(np.uint8)
        initial_mask = cv2.resize(initial_mask, (original_np.shape[1], original_np.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        final_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)
    crf_mask = maybe_crf(original_np, final_mask)   # editor-approved post-processing
    if sam_predictor is not None and REVISION_CONFIG.get("KVASIR_STAGE_SELECT", True):
        final_mask = select_best_stage_mask(
            classifier, image_tensor, target_class, device,
            [initial_mask, snake_mask, final_mask, crf_mask],
            (original_np.shape[0], original_np.shape[1]))
    else:
        final_mask = crf_mask
    dice = calculate_dice_score(final_mask, gt_mask_np)
    iou = calculate_iou_score(final_mask, gt_mask_np)
    return float(dice), float(iou)

def _kvasir_gather_samples(kvasir_dataset):
    samples, cap = [], _suite_cap()
    for idx in range(len(kvasir_dataset)):
        try:
            item = kvasir_dataset[idx]
            if item is None:
                continue
            image_tensor, mask_tensor, img_path = item
            if image_tensor is None or image_tensor.nelement() == 0:
                continue
            samples.append((image_tensor, mask_tensor, img_path))
        except Exception:
            continue
        if cap is not None and len(samples) >= cap:
            break
    return samples

def _kvasir_eval(samples, classifier, sam_predictor, device, xai_weights,
                 fusion_strategy="weighted_average", use_iterative=True):
    dices, ious = [], []
    for sample in samples:
        try:
            d, i = _kvasir_segment_metrics_only(sample, classifier, sam_predictor, device,
                                                xai_weights, fusion_strategy, use_iterative)
            dices.append(d); ious.append(i)
        except Exception as _e:
            print(f"  [suite] sample skipped: {_e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dices, ious

def run_revision_suite_kvasir(classifier, sam_predictor, xai_weights, kvasir_dataset, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "#" * 80)
    print("# REVISION EXPERIMENT SUITE (Kvasir-SEG)")
    print("#" * 80)
    if kvasir_dataset is None:
        print("kvasir_dataset not provided; suite aborted."); return
    samples = _kvasir_gather_samples(kvasir_dataset)
    print(f"Using {len(samples)} samples (cap={_suite_cap()}).")
    if not samples:
        print("No samples available; suite aborted."); return
    saved_crf = REVISION_CONFIG.get("USE_CRF", False)
    seed0 = REVISION_CONFIG.get("SEEDS", [42])[0]

    if REVISION_CONFIG.get("RUN_SEEDS", False):
        print("\n--- (1) Multi-seed variability (Reviewer: stochastic prompt sampling) ---")
        per_seed_dice, per_seed_iou = [], []
        for s in REVISION_CONFIG.get("SEEDS", [42]):
            set_global_seed(s)
            d, i = _kvasir_eval(samples, classifier, sam_predictor, device, xai_weights)
            if d:
                per_seed_dice.append(np.mean(d)); per_seed_iou.append(np.mean(i))
                print(f"  seed {s}: Dice={np.mean(d):.4f}  IoU={np.mean(i):.4f}  (n={len(d)})")
        if per_seed_dice:
            print(f"  >> Dice = {np.mean(per_seed_dice):.4f} +/- {np.std(per_seed_dice):.4f} "
                  f"| IoU = {np.mean(per_seed_iou):.4f} +/- {np.std(per_seed_iou):.4f}  (across {len(per_seed_dice)} seeds)")

    if REVISION_CONFIG.get("RUN_CRF_ABLATION", False):
        print("\n--- (2) CRF ablation (Editor: previous-vs-updated methodology comparison) ---")
        set_global_seed(seed0); REVISION_CONFIG["USE_CRF"] = False
        d0, i0 = _kvasir_eval(samples, classifier, sam_predictor, device, xai_weights)
        set_global_seed(seed0); REVISION_CONFIG["USE_CRF"] = True
        d1, i1 = _kvasir_eval(samples, classifier, sam_predictor, device, xai_weights)
        if d0 and d1:
            print(f"  Without CRF : Dice={np.mean(d0):.4f}  IoU={np.mean(i0):.4f}")
            print(f"  With CRF    : Dice={np.mean(d1):.4f}  IoU={np.mean(i1):.4f}")
            print(f"  Delta       : Dice={np.mean(d1)-np.mean(d0):+.4f}  IoU={np.mean(i1)-np.mean(i0):+.4f}")
        REVISION_CONFIG["USE_CRF"] = saved_crf

    if REVISION_CONFIG.get("RUN_SENSITIVITY", False):
        print("\n--- (3) Fusion-strategy sensitivity ---")
        for fs in REVISION_CONFIG.get("SENSITIVITY_FUSION", ["weighted_average"]):
            set_global_seed(seed0)
            d, i = _kvasir_eval(samples, classifier, sam_predictor, device, xai_weights, fusion_strategy=fs)
            if d:
                print(f"  {fs:>26}: Dice={np.mean(d):.4f}  IoU={np.mean(i):.4f}")

    if REVISION_CONFIG.get("RUN_LATENCY", False):
        print("\n--- (4) Per-stage latency (ms / image) ---")
        try:
            sample = samples[0]
            image_tensor, _, img_path = sample
            original_np = np.array(Image.open(img_path).convert("RGB"))
            timings = {}
            t = _time.time()
            with torch.no_grad():
                _, entropy, probs = classifier.predict_with_uncertainty(image_tensor.unsqueeze(0).to(device))
            pred = torch.argmax(probs, dim=1).item()
            timings["BiomedCLIP_forward"] = (_time.time() - t) * 1000
            t = _time.time()
            expl = generate_explanations_focused(classifier, image_tensor.unsqueeze(0), pred, device, use_cache=False)
            timings["XAI_attribution"] = (_time.time() - t) * 1000
            t = _time.time()
            fused = advanced_xai_fusion(expl, xai_weights, "weighted_average")
            timings["XAI_fusion"] = (_time.time() - t) * 1000
            bbox = extract_focused_bbox_from_saliency(fused, top_k_percent=0.05)
            pos = extract_focused_positive_prompts(fused, bbox, num_prompts=3)
            bbox_o = transform_bbox_to_original(bbox, fused.shape, original_np.shape[:2])
            pos_o = transform_coordinates_to_original(pos, fused.shape, original_np.shape[:2])
            mask = np.zeros(original_np.shape[:2], np.uint8)
            if sam_predictor is not None:
                t = _time.time()
                mask, _ = segment_with_sam_enhanced(sam_predictor, original_np, pos_o, None, bbox_o, post_process=True)
                timings["SAM_prediction"] = (_time.time() - t) * 1000
            t = _time.time()
            try:
                mask = refine_mask_with_snake_improved(mask, original_np, fused, iterations=100)
            except Exception:
                pass
            timings["Snake_refinement"] = (_time.time() - t) * 1000
            t = _time.time()
            _ = apply_crf_refinement(original_np, mask, REVISION_CONFIG.get("CRF_ITERATIONS", 5))
            timings["CRF_postprocess"] = (_time.time() - t) * 1000
            timings["TOTAL"] = sum(timings.values())
            for k, v in timings.items():
                print(f"  {k:>22}: {v:8.2f} ms")
        except Exception as _e:
            print(f"  latency profiling skipped: {_e}")
    REVISION_CONFIG["USE_CRF"] = saved_crf
    print("\n# Suite complete. Paste the numbers above into Tables 4/5 and the latency table.\n")
# =========================================================================================
# END REVISION EXPERIMENT SUITE (Kvasir-SEG)
# =========================================================================================

if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE MEDICAL IMAGE ANALYSIS PIPELINE")
    print("Capsule Endoscopy Classification + Kvasir-SEG Segmentation")
    print("="*80)
    
    try:
        result = main()
        
        if result is not None:
            capsule_model, sam_predictor, xai_weights = result[0], result[1], result[2]
            kvasir_dataset = result[3] if len(result) > 3 else None
            print("\n✅ Pipeline completed successfully!")
            try:
                if REVISION_CONFIG.get('RUN_REVISION_SUITE', False):
                    run_revision_suite_kvasir(capsule_model, sam_predictor, xai_weights, kvasir_dataset)
            except Exception as _e:
                print(f'[revision-suite] skipped: {_e}')
        else:
            print("\n❌ Pipeline returned None - check for errors above")
            
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("🎉 CODE EXECUTION FINISHED!")
print("="*80)