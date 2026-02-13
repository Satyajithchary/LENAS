
def normalize_xai_map(xai_map, method='minmax'):
    """Normalize XAI map."""
    if method == 'minmax':
        map_min, map_max = xai_map.min(), xai_map.max()
        if map_max > map_min:
            return (xai_map - map_min) / (map_max - map_min)
        else:
            return np.zeros_like(xai_map)

def advanced_xai_fusion(explanations, weights, fusion_strategy='weighted_average'):
    """Focused fusion of XAI maps."""
    normalized_maps = {}
    for method, xai_map in explanations.items():
        normalized_maps[method] = normalize_xai_map(xai_map, 'minmax')

    if fusion_strategy == 'weighted_average':
        fused_map = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * norm_map
            total_weight += weight
        if total_weight > 0:
            fused_map /= total_weight

    elif fusion_strategy == 'multiplicative_consensus':
        fused_map = np.ones_like(next(iter(normalized_maps.values())))
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map *= (norm_map + 0.01) ** weight

    elif fusion_strategy == 'weighted_geometric_mean':
        log_sum = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            log_sum += weight * np.log(norm_map + 1e-8)
            total_weight += weight
        fused_map = np.exp(log_sum / total_weight) if total_weight > 0 else log_sum

    elif fusion_strategy == 'rank_aggregation':
        rank_maps = {}
        for method, norm_map in normalized_maps.items():
            flat_map = norm_map.flatten()
            ranks = np.argsort(np.argsort(flat_map)) / len(flat_map)
            rank_maps[method] = ranks.reshape(norm_map.shape)

        fused_map = np.zeros_like(next(iter(rank_maps.values())))
        for method, rank_map in rank_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * rank_map
    else:
        fused_map = np.zeros_like(next(iter(normalized_maps.values())))
        total_weight = 0
        for method, norm_map in normalized_maps.items():
            weight = weights.get(method, 0.33)
            fused_map += weight * norm_map
            total_weight += weight
        if total_weight > 0:
            fused_map /= total_weight

    return fused_map
