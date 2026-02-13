
def refine_mask_with_snake_improved(mask, image_np, fused_map, iterations=100,
                                   alpha=0.05, beta=5, gamma=0.001, max_deviation=30):
    """
    Snake refinement with saliency constraints.

    Improvements:
    - Weight edges by XAI saliency
    - Reject if snake deviates too far
    - Reduced iterations for speed
    """
    if mask.sum() == 0:
        print("  Empty mask, skipping snake refinement")
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
            print("  Invalid contour shape, skipping snake")
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
            beta=beta,          # Lower smoothness (allows irregular shapes)
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
            print(f" Snake deviated too far ({deviation:.1f}px > {max_deviation}px), keeping original")
            return mask

        # Post-process
        refined_mask = post_process_mask(refined_mask, image_np, min_area=100, hole_area=50)

        print(f"  Snake refinement complete (deviation: {deviation:.1f}px)")
        return refined_mask

    except Exception as e:
        print(f" Snake algorithm failed: {e}, returning original mask")
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
