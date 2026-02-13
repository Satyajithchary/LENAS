
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
