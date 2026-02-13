
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


def calculate_mask_quality_score_improved(mask, image_np, fused_map):
    """
    Calculate mask quality with better metrics.

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
