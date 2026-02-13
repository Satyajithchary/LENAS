
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

    # Generate XAI
    explanations = generate_explanations_focused(classifier, image_tensor.unsqueeze(0),
                                                predicted_class, device, use_cache=True)
    fused_map = advanced_xai_fusion(explanations, xai_weights, fusion_strategy)

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
        initial_mask, sam_score = segment_with_sam_enhanced(
            sam_predictor, original_np, positive_prompts_orig,
            negative_prompts_orig, bbox_orig, post_process=True
        )
        snake_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)

        # Run iterative self-correction
        final_mask, iter_history = iterative_self_correction_improved(
            classifier, sam_predictor, original_np, image_tensor, predicted_class,
            device, xai_weights, max_iterations=3
        )
    else:
        # Fallback segmentation
        threshold = np.percentile(fused_map, 95)
        initial_mask = (fused_map >= threshold).astype(np.uint8)
        initial_mask = cv2.resize(initial_mask, (original_np.shape[1], original_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        snake_mask = refine_mask_with_snake_improved(initial_mask, original_np, fused_map, iterations=100)
        final_mask, sam_score, iter_history = snake_mask, 0.0, None

    # Calculate metrics
    dice_score = calculate_dice_score(final_mask, gt_mask_np)
    iou_score = calculate_iou_score(final_mask, gt_mask_np)

    print(f" Segmentation Metrics:")
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
    plt.show()

    return final_mask, dice_score, iou_score, iter_history
