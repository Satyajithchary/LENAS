
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
