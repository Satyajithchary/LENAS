
class DifferentialBiomedCLIP(nn.Module):
    """
    Hybrid model with contrastive learning + classification head
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

        self.model = base_model

        # Contrastive learning components
        if self.use_contrastive:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

            if class_names is None:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            self.class_names = class_names
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