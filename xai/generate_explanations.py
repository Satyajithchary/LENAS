from .xai_cache import xai_cache

def generate_explanations_focused(model, x_batch, y_batch, device, use_cache=True):
    """Generate FOCUSED explanations with caching."""
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

    # Saliency with absolute values
    saliency = Saliency(model)
    sal_attr = saliency.attribute(x_batch_grad, target=target_tensor.to(device), abs=True)
    explanations['Saliency'] = sal_attr.sum(axis=1).squeeze().cpu().detach().numpy()

    # Integrated Gradients (reduced steps for speed)
    ig = IntegratedGradients(model)
    ig_attr = ig.attribute(x_batch_grad, target=target_tensor.to(device), n_steps=30)
    explanations['IntegratedGradients'] = np.abs(ig_attr.sum(axis=1).squeeze().cpu().detach().numpy())

    # GradientShap (reduced samples for speed)
    gs = GradientShap(model)
    baseline = torch.zeros_like(x_batch_grad)
    gs_attr = gs.attribute(x_batch_grad, baselines=baseline, target=target_tensor.to(device), n_samples=15)
    explanations['GradientShap'] = np.abs(gs_attr.sum(axis=1).squeeze().cpu().detach().numpy())

    # Cache the result
    if use_cache:
        xai_cache.set(x_batch[0], label_for_cache, explanations)

    return explanations
