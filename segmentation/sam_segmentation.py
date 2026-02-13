
from xai.generate_explanations import generate_explanations_focused
from xai.xai_fusion import advanced_xai_fusion
from xai.prompt_selection import uncertainty_guided_prompts


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

    smoothed_roi = ndimage.gaussian_filter(roi, sigma=2.0)

    # Uses very high threshold - only top 10% within ROI
    threshold_val = np.percentile(smoothed_roi, 90)

    # Find local maxima
    local_maxima = peak_local_max(
        smoothed_roi,
        min_distance=min_distance,
        threshold_abs=threshold_val,
        num_peaks=num_prompts * 2 
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
