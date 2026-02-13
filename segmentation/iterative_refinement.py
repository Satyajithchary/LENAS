
def iterative_self_correction_improved(classifier, sam_predictor, image_np, image_tensor,
                                      original_label, device, xai_weights,
                                      max_iterations=3, convergence_threshold=0.02,
                                      confidence_threshold=0.15):

    print("\n" + "="*70)
    print("IMPROVED ITERATIVE SELF-CORRECTION LOOP")
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

    # Initial SAM segmentation
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

    # Apply improved snake
    print("  Applying improved Snake refinement...")
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
    no_improvement_count = 0  

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

                    #  Validate prompts before adding
                    valid_prompts = validate_prompt_additions(additional_prompts, current_mask,
                                                             fused_resized, min_saliency_threshold=0.7)

                    if len(valid_prompts) > 0:
                        positive_prompts_orig = np.vstack([positive_prompts_orig, valid_prompts])
                        print(f"  ➕ Added {len(valid_prompts)} validated prompts")
        else:
            print(f" Conf diff is good, fine-tuning only...")

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

        print("  Applying improved Snake refinement...")
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

        # STRICTER QUALITY GATE: Accepts only if BOTH improve AND better than best seen
        improvement = (
            confidence_diff_new > confidence_diff and
            mask_quality_new > mask_quality and
            confidence_diff_new > best_confidence_diff and
            mask_quality_new > best_quality
        )

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
            no_improvement_count = 0 
        else:
            print(f" Rejected! Quality gate not passed, reverting to best")
            # Revert to best mask
            current_mask = best_mask.copy()
            no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= 2:
                print(f" Early stopping: No improvement for 2 iterations")
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



def validate_prompt_additions(additional_prompts, current_mask, fused_map, min_saliency_threshold=0.7):
    """
    Validate that new prompts are meaningful before adding them.

    Returns:
        valid_prompts: Array of validated prompts
    """
    if len(additional_prompts) == 0:
        return additional_prompts

    fused_norm = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min() + 1e-8)

    valid_prompts = []
    for prompt in additional_prompts:
        x, y = int(prompt[0]), int(prompt[1])

        # Checks if within bounds
        if 0 <= y < fused_norm.shape[0] and 0 <= x < fused_norm.shape[1]:
            # Checks saliency at prompt location
            saliency_value = fused_norm[y, x]

            # Only adds if saliency is high enough
            if saliency_value >= min_saliency_threshold:
                valid_prompts.append(prompt)
            else:
                print(f"    ❌ Rejected prompt at ({x},{y}) - low saliency: {saliency_value:.3f}")

    return np.array(valid_prompts) if valid_prompts else np.array([])



def evaluate_masked_confidence(classifier, image_tensor, mask, label, device, original_shape):
    """Evaluate classifier confidence on masked regions."""
    tensor_h, tensor_w = image_tensor.shape[1], image_tensor.shape[2]
    mask_resized = cv2.resize(mask, (tensor_w, tensor_h), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_resized).float().to(device)
    image_tensor = image_tensor.to(device)

    # Masked-in: Keeps pathology
    image_masked_in = image_tensor.clone()
    for c in range(3):
        image_masked_in[c] = image_masked_in[c] * mask_tensor

    # Masked-out: Keeps background
    image_masked_out = image_tensor.clone()
    for c in range(3):
        image_masked_out[c] = image_masked_out[c] * (1 - mask_tensor)

    with torch.no_grad():
        _, _, probs_in = classifier.predict_with_uncertainty(image_masked_in.unsqueeze(0))
        _, _, probs_out = classifier.predict_with_uncertainty(image_masked_out.unsqueeze(0))
        conf_in = probs_in[0][label].item()
        conf_out = probs_out[0][label].item()

    return conf_in, conf_out
