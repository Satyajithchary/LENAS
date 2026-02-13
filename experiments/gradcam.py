def generate_gradcam_heatmap(model, input_tensor, target_class, device, use_cuda=torch.cuda.is_available()):
    """
    Hook last conv layer (model.features[-3] is conv before AdaptiveAvgPool in our SimpleCNN).
    Returns heatmap resized to input HxW (numpy float32).
    """
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    # find last conv within model.features
    target_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target_conv = m
    if target_conv is None:
        raise RuntimeError("No Conv2d found in model for Grad-CAM hooks")

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    handle_f = target_conv.register_forward_hook(forward_hook)
    handle_b = target_conv.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    handle_f.remove(); handle_b.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks failed")

    weights = gradients.mean(dim=(2,3), keepdim=True)  # global average pool over spatial dims
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    # Upsample to input dims
    _, H, W = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3] if input_tensor.dim() >=4 else (None,None)
    cam_resized = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
    return cam_resized

def evaluate_gradcam(dataset, model, device, threshold_percentile=75, num_samples=None):
    """
    For each sample generate grad-cam, threshold at percentile, compute dice/iou.
    """
    results = []
    n = len(dataset)
    indices = np.arange(n)
    if num_samples is not None:
        indices = np.random.choice(indices, min(num_samples, n), replace=False)
    for idx in indices:
        img_tensor, mask_tensor, path = dataset[idx]
        # predict class using model (if multi-class pick argmax)
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred_cls = torch.argmax(logits, dim=1).item()
        cam = generate_gradcam_heatmap(model, img_tensor, pred_cls, device)
        thresh = np.percentile(cam, threshold_percentile)
        pred_mask = (cam >= thresh).astype(np.uint8)
        # if cam smaller than original spatial dims, resize to original image dims
        img_np = np.array(Image.open(path).convert("RGB"))
        pred_mask = cv2.resize(pred_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = tensor_to_numpy_mask(mask_tensor)
        dice = calculate_dice_score(pred_mask, gt)
        iou = calculate_iou_score(pred_mask, gt)
        results.append((dice, iou))
    return results

