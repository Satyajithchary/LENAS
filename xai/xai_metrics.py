
def calculate_faithfulness_quantus(model, image_tensor, label, explanations, device, subset_size=224, nr_runs=10):
    """Faithfulness via perturbation correlation."""
    model.eval()
    faithfulness_scores = {}

    with torch.no_grad():
        original_logits = model(image_tensor.unsqueeze(0).to(device))
        original_pred = torch.softmax(original_logits, dim=1)[0, label].item()

    for method, xai_map in explanations.items():
        correlations = []

        for run in range(nr_runs):
            flat_indices = np.random.choice(xai_map.size, min(subset_size, xai_map.size), replace=False)
            importance_scores = xai_map.flatten()[flat_indices]

            perturbed_image = image_tensor.clone()
            coords = np.unravel_index(flat_indices, xai_map.shape)
            perturbed_image[:, coords[0], coords[1]] = 0.0

            with torch.no_grad():
                perturbed_logits = model(perturbed_image.unsqueeze(0).to(device))
                perturbed_pred = torch.softmax(perturbed_logits, dim=1)[0, label].item()

            pred_drop = original_pred - perturbed_pred

            if len(np.unique(importance_scores)) > 1:
                correlation = np.corrcoef(importance_scores, [pred_drop] * len(importance_scores))[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)

        faithfulness_scores[method] = np.mean(correlations) if correlations else 0.0

    return faithfulness_scores

def calculate_robustness_quantus(model, image_tensor, label, device, nr_samples=10, noise_level=0.2):
    """Robustness to input perturbations."""
    base_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), label, device)
    robustness_scores = {}
    similarities_per_method = {method: [] for method in base_explanations.keys()}

    for _ in range(nr_samples):
        noise = torch.rand_like(image_tensor) * 2 * noise_level - noise_level
        noisy_image = torch.clamp(image_tensor + noise, 0, 1)
        noisy_explanations = generate_explanations_focused(model, noisy_image.unsqueeze(0), label, device)

        for method in base_explanations.keys():
            base_map = base_explanations[method]
            noisy_map = noisy_explanations[method]

            base_norm = (base_map - base_map.min()) / (base_map.max() - base_map.min() + 1e-8)
            noisy_norm = (noisy_map - noisy_map.min()) / (noisy_map.max() - noisy_map.min() + 1e-8)

            similarity = ssim(base_norm, noisy_norm, data_range=1.0)
            if not np.isnan(similarity):
                similarities_per_method[method].append(similarity)

    for method in similarities_per_method:
        robustness_scores[method] = np.mean(similarities_per_method[method]) if similarities_per_method[method] else 0.0

    return robustness_scores

def calculate_sparseness_quantus(explanations):
    """Sparseness using Gini coefficient."""
    sparseness_scores = {}
    for method, xai_map in explanations.items():
        abs_map = np.abs(xai_map.flatten())
        if abs_map.sum() == 0:
            sparseness_scores[method] = 1.0
            continue

        abs_map = abs_map / abs_map.sum()
        n = len(abs_map)
        sorted_abs = np.sort(abs_map)
        cumsum_sorted = np.cumsum(sorted_abs)
        gini = (n + 1 - 2 * np.sum(cumsum_sorted)) / n
        sparseness_scores[method] = gini

    return sparseness_scores

def calculate_localization_quantus(explanations):
    """Localization using relevance rank accuracy."""
    localization_scores = {}

    for method, xai_map in explanations.items():
        threshold = np.percentile(np.abs(xai_map), 90)
        relevant_mask = np.abs(xai_map) >= threshold

        total_relevant = relevant_mask.sum()
        if total_relevant == 0:
            localization_scores[method] = 0.0
        else:
            sorted_indices = np.argsort(np.abs(xai_map.flatten()))[::-1]
            top_k = int(0.1 * len(sorted_indices))

            relevant_in_top_k = 0
            for idx in sorted_indices[:top_k]:
                coords = np.unravel_index(idx, xai_map.shape)
                if relevant_mask[coords]:
                    relevant_in_top_k += 1

            localization_scores[method] = relevant_in_top_k / top_k if top_k > 0 else 0.0

    return localization_scores

def calculate_randomization_quantus(model, image_tensor, label, device, num_classes):
    """Randomization test."""
    correct_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), label, device)
    random_label = np.random.choice([i for i in range(num_classes) if i != label])
    random_explanations = generate_explanations_focused(model, image_tensor.unsqueeze(0), random_label, device)

    randomization_scores = {}
    for method in correct_explanations.keys():
        correct_map = correct_explanations[method]
        random_map = random_explanations[method]

        correct_norm = (correct_map - correct_map.min()) / (correct_map.max() - correct_map.min() + 1e-8)
        random_norm = (random_map - random_map.min()) / (random_map.max() - random_map.min() + 1e-8)

        similarity = ssim(correct_norm, random_norm, data_range=1.0)
        dissimilarity = 1 - similarity if not np.isnan(similarity) else 1.0
        randomization_scores[method] = max(0.0, dissimilarity)

    return randomization_scores

def evaluate_xai_methods_quantus(model, dataloader, device, num_classes, num_samples=20, CLASS_LABELS=None):
    """Quantus-style evaluation."""
    print("Starting Quantus-style XAI evaluation...")

    if len(dataloader.dataset) < num_samples:
        num_samples = len(dataloader.dataset)

    subset_indices = np.random.choice(len(dataloader.dataset), num_samples, replace=False)
    subset_loader = DataLoader(dataloader.dataset, batch_size=1,
                              sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

    methods = ['Saliency', 'IntegratedGradients', 'GradientShap']
    results = {m: {'faithfulness': [], 'robustness': [], 'sparseness': [],
                   'localization': [], 'randomization': []} for m in methods}

    for images, labels, _ in tqdm(subset_loader, desc="Evaluating XAI"):
        image, label = images.squeeze(0), labels.item()

        explanations = generate_explanations_focused(model, image.unsqueeze(0), label, device)

        faithfulness = calculate_faithfulness_quantus(model, image, label, explanations, device)
        robustness = calculate_robustness_quantus(model, image, label, device)
        sparseness = calculate_sparseness_quantus(explanations)
        localization = calculate_localization_quantus(explanations)
        randomization = calculate_randomization_quantus(model, image, label, device, num_classes)

        for method in methods:
            results[method]['faithfulness'].append(faithfulness.get(method, 0))
            results[method]['robustness'].append(robustness.get(method, 0))
            results[method]['sparseness'].append(sparseness.get(method, 0))
            results[method]['localization'].append(localization.get(method, 0))
            results[method]['randomization'].append(randomization.get(method, 0))

        gc.collect()
        torch.cuda.empty_cache()

    avg_scores = {m: {k: np.mean(v) for k, v in scores.items()} for m, scores in results.items()}

    print("\n--- Average XAI Metric Scores ---")
    df = pd.DataFrame.from_dict(avg_scores, orient='index')
    print(df.round(4))

    df_norm = df.copy()
    for col in df_norm.columns:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max > col_min:
            if col == 'robustness':
                df_norm[col] = 1 - (df_norm[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 1.0

    method_scores = df_norm.mean(axis=1)
    total_score = method_scores.sum()
    xai_weights = (method_scores / total_score).to_dict() if total_score > 0 else {m: 1/len(methods) for m in methods}

    print("\n--- Derived XAI Weights ---")
    for method, weight in xai_weights.items():
        print(f"{method}: {weight:.4f}")

    return xai_weights, avg_scores
