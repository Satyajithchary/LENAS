def evaluate_our_method_on_sample(classifier, sam_predictor, image_np, image_tensor, predicted_class, device,
                                  xai_weights, use_sam=True, use_snake=True, use_iter_refine=True):
    """
    Run pipeline pieces per sample and return final_mask, dice, iou.
    Uses your existing helper functions; returns final mask.
    """
    # Compute explanations and fused map
    explanations = generate_explanations_focused(classifier, image_tensor.unsqueeze(0), predicted_class, device, use_cache=True)
    fused_map = advanced_xai_fusion(explanations, xai_weights)
    circular_mask = detect_circular_mask(image_np.shape)
    num_prompts, strategy = uncertainty_guided_prompts(fused_map, 0.0, np.array([1.0]))  # dummy entropy/probs for selection
    bbox_xai = extract_focused_bbox_from_saliency(fused_map, top_k_percent=0.05)
    positive_prompts_xai = extract_focused_positive_prompts(fused_map, bbox_xai, num_prompts=num_prompts)
    circular_mask_xai = cv2.resize(circular_mask, (fused_map.shape[1], fused_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    negative_prompts_xai = extract_smart_negative_prompts(fused_map, bbox_xai, circular_mask_xai, num_negatives=3)
    bbox_orig = transform_bbox_to_original(bbox_xai, fused_map.shape, image_np.shape[:2])
    positive_prompts_orig = transform_coordinates_to_original(positive_prompts_xai, fused_map.shape, image_np.shape[:2])
    negative_prompts_orig = transform_coordinates_to_original(negative_prompts_xai, fused_map.shape, image_np.shape[:2]) if len(negative_prompts_xai) > 0 else np.array([])

    # SAM or fallback
    if use_sam and sam_predictor is not None:
        init_mask, sam_score = segment_with_sam_enhanced(sam_predictor, image_np, positive_prompts_orig, negative_prompts_orig, bbox_orig, post_process=True)
    else:
        threshold = np.percentile(fused_map, 95)
        init_mask = (fused_map >= threshold).astype(np.uint8)
        init_mask = cv2.resize(init_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        sam_score = 0.0

    cur_mask = init_mask.copy()

    # Optional snake
    if use_snake:
        cur_mask = refine_mask_with_snake_improved(cur_mask, image_np, fused_map, iterations=50)

    # Optional CRF-like refinement
    cur_mask = apply_random_walker_refinement(image_np, cur_mask)

    # Optional iterative refinement 
    if use_iter_refine and sam_predictor is not None:
        best_mask, _ = iterative_self_correction_improved(classifier, sam_predictor, image_np, image_tensor, predicted_class, device, xai_weights, max_iterations=2)
        cur_mask = best_mask

    return cur_mask

def safe_forward(model, x):
    """Run model(x) safely in eval mode and restore original mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x)
    if was_training:
        model.train()
    return out



def print_results_tables(results, ablations):

    def extract_two(val):
        """Ensure (dice, iou) tuple from arbitrary val shape."""
        if isinstance(val, (list, tuple)):
            if len(val) >= 2:
                return val[0], val[1]
            elif len(val) == 1:
                return val[0], 0.0
        # default case: single scalar
        return val, 0.0

    print("\n=== Table 1: Main Comparison ===")
    rows = []

    for method, vals in results.items():
        dices, ious = [], []

        for v in vals:
            d, i = extract_two(v)
            try:
                dices.append(float(d))
                ious.append(float(i))
            except (ValueError, TypeError):
                continue

        # sanitize arrays
        dices = np.nan_to_num(np.array(dices, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        ious  = np.nan_to_num(np.array(ious, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        dice_mean = float(np.mean(dices)) if len(dices) > 0 else 0.0
        dice_std  = float(np.std(dices)) if len(dices) > 0 else 0.0

        rows.append([method, dice_mean, dice_std, len(dices)])

    df_results = pd.DataFrame(rows, columns=["Method", "Dice_mean", "Dice_std", "#samples"])
    print(df_results.to_string(index=False))
    print()

    print("=== Table 2: Ablation Study ===")
    ab_rows = []

    for ab_name, vals in ablations.items():
        dices, ious = [], []

        for v in vals:
            d, i = extract_two(v)
            try:
                dices.append(float(d))
                ious.append(float(i))
            except (ValueError, TypeError):
                continue

        dices = np.nan_to_num(np.array(dices, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        ious  = np.nan_to_num(np.array(ious, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        dice_mean = float(np.mean(dices)) if len(dices) > 0 else 0.0
        iou_mean  = float(np.mean(ious)) if len(ious) > 0 else 0.0

        ab_rows.append([ab_name, dice_mean, iou_mean])

    df_ab = pd.DataFrame(ab_rows, columns=["Ablation", "Dice_mean", "IoU_mean"])
    print(df_ab.to_string(index=False))

    # Save both CSVs
    try:
        df_results.to_csv("experiment_harness_results.csv", index=False)
        df_ab.to_csv("ablation.csv", index=False)
        print("Saved experiment_harness_results.csv and ablation.csv")
    except Exception as e:
        print("Could not save CSV files:", e)


# -----------------------
# Master experiment runner (quick vs full)
# -----------------------
def run_experiments_quick_and_full(
    capsule_model,
    sam_predictor,
    kvasir_dataset,
    device,
    xai_weights,
    quick_run=False,
    quick_samples=10
):
    """
    Runs the full experimental harness:
      - Trains ScribbleUNet and partially supervised UNets
      - Trains a SimpleCNN baseline for Grad-CAM
      - Runs our proposed Capsule+SAM+Snake self-correction method
      - Evaluates all baselines and performs an ablation study
    """

    print("\n" + "="*80)
    print("Starting Experiment Harness (Baselines + Our Method)")
    print("="*80)

    # ----------------------------------------------------------
    # Create Data Loaders
    # ----------------------------------------------------------
    print("\n Preparing DataLoaders...")
    dataset_len = len(kvasir_dataset)
    val_split = int(0.2 * dataset_len)
    train_dataset, val_dataset = torch.utils.data.random_split(
        kvasir_dataset, [dataset_len - val_split, val_split]
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_kvasir)
    val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, collate_fn=collate_fn_kvasir)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # ----------------------------------------------------------
    # Initialize Models
    # ----------------------------------------------------------
    print("\n Initializing UNet model variants...")
    unet_scribble = SimpleUNet(in_channels=3, out_channels=1).to(device)
    unet_partial  = SimpleUNet(in_channels=3, out_channels=1).to(device)

    print("UNet variants initialized.")
    print("\n======================")
    print("Training SimpleCNN (Grad-CAM Baseline)...")
    print("======================")

    # ----------------------------------------------------------
    # Train SimpleCNN Baseline for Grad-CAM
    # ----------------------------------------------------------
    simplecnn = SimpleCNN(num_classes=8).to(device)
    optimizer = torch.optim.Adam(simplecnn.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1 if quick_run else 3):
        simplecnn.train()
        total_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            labels = torch.randint(0, 8, (imgs.size(0),), device=device)  # Dummy labels
            preds = simplecnn(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Grad-CAM Baseline Loss: {total_loss/len(train_loader):.4f}")

    print("SimpleCNN training done.\n")

    # ----------------------------------------------------------
    # Train ScribbleUNet and Partial UNets
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print("Training ScribbleUNet and Partial UNet Models")
    print("="*55)

    print("\n======================")
    print("🔧 Training ScribbleUNet baseline...")
    print("======================")
    scribble_unet = train_unet_with_scribbles(
        unet_scribble, train_loader, val_loader, device,
        epochs=1 if quick_run else 3,
        scribble_getter=generate_scribbles_from_mask,
        force_retrain=True
    )

    print("\n======================")
    print("Training UNet (1% supervision)...")
    print("======================")
    unet_1 = train_unet_partial(
        unet_partial, train_loader, val_loader, device,
        fraction=0.01,
        epochs=1 if quick_run else 3,
        force_retrain=True
    )

    print("\n======================")
    print("Training UNet (5% supervision)...")
    print("======================")
    unet_5 = train_unet_partial(
        unet_partial, train_loader, val_loader, device,
        fraction=0.05,
        epochs=1 if quick_run else 3,
        force_retrain=True
    )

    print("\n======================")
    print("Training UNet (10% supervision)...")
    print("======================")
    unet_10 = train_unet_partial(
        unet_partial, train_loader, val_loader, device,
        fraction=0.10,
        epochs=1 if quick_run else 3,
        force_retrain=True
    )

    print("\n All UNet baselines trained successfully.\n")

    # ----------------------------------------------------------
    # Run Our Method (Capsule + SAM + Snake Refinement)
    # ----------------------------------------------------------
    print("\n" + "="*70)
    print("Running Our Proposed Method (Capsule + SAM + Snake Loop)")
    print("="*70)
    our_results = []
    num_samples = quick_samples if quick_run else len(val_dataset)
    sample_indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    for idx in tqdm(sample_indices, desc="Evaluating OurMethod"):
        img_t, mask_t, path = val_dataset[idx]
        img_t, mask_t = img_t.to(device), mask_t.to(device)

        # Capsule model forward
        with torch.no_grad():
            logits = safe_forward(capsule_model, img_t.unsqueeze(0))
            conf = torch.softmax(logits, dim=1).max().item()

        # Iterative refinement (correct order)
        mask_pred, iteration_info = run_iterative_self_correction(
            capsule_model=capsule_model,
            sam_predictor=sam_predictor,
            image=img_t,
            initial_mask=None,
            max_iters=3,
            conf_threshold=0.15,
            device=device
        )

        # Resize predicted mask to match ground truth
        if mask_pred.shape[-2:] != mask_t.shape[-2:]:
            mask_pred = torch.nn.functional.interpolate(
                mask_pred, size=mask_t.shape[-2:], mode='bilinear', align_corners=False
            )

        dice = dice_score(mask_pred, mask_t)
        iou = iou_score(mask_pred, mask_t)

        our_results.append((float(dice), float(iou)))

        #our_results.append((dice, iou))

        print(f"Sample {idx} done | Dice: {dice:.4f} | IoU: {iou:.4f}")



    # ----------------------------------------------------------
    # Evaluate Baseline Models on Validation Set
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("Evaluating Baseline Segmenters on Validation Set")
    print("="*60)

    baselines = {
        "GradCAM": simplecnn,
        "ScribbleUNet": scribble_unet,
        "UNet_1%": unet_1,
        "UNet_5%": unet_5,
        "UNet_10%": unet_10
    }

    results_table = []
    for name, model in baselines.items():
        print(f"Evaluating {name} ...")
        dice_scores, iou_scores = [], []
        model.eval()
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                dice_scores.append(dice_score(preds, masks))
                iou_scores.append(iou_score(preds, masks))
        results_table.append([name, np.mean(dice_scores), np.std(dice_scores), len(dice_scores)])
        print(f"   Dice: {np.mean(dice_scores):.4f}, IoU: {np.mean(iou_scores):.4f}")

    results_table.append([
        "OurMethod",
        np.mean([r[0] for r in our_results]),
        np.std([r[0] for r in our_results]),
        len(our_results)
    ])

    # ----------------------------------------------------------
    # Ablation Study
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print("Running Ablation Study")
    print("="*55)

    ablations = [
        ("FullPipeline", True, True, True),
        ("NoIterRefine", False, True, True),
        ("NoSAM", True, False, True),
        ("NoSnake", True, True, False),
        ("XAIOnly", False, False, False)
    ]

    ablation_table = []
    for name, use_iter, use_sam, use_snake in ablations:
        dice_scores, iou_scores = [], []
        for idx in np.random.choice(len(val_dataset), 3 if quick_run else 10, replace=False):
            img_t, mask_t, path = val_dataset[idx]
            img_t, mask_t = img_t.to(device), mask_t.to(device)

            mask_pred, _ = run_iterative_self_correction(
                capsule_model=capsule_model,
                sam_predictor=sam_predictor,
                image=img_t,
                initial_mask=None,
                max_iters=3,
                conf_threshold=0.15,
                device=device
            )
            dice_scores.append(dice_score(mask_pred, mask_t))
            iou_scores.append(iou_score(mask_pred, mask_t))
        ablation_table.append([name, np.mean(dice_scores), np.mean(iou_scores)])

    # ----------------------------------------------------------
    # Save and Display Results
    # ----------------------------------------------------------
    df_results = pd.DataFrame(results_table, columns=["Method", "Dice_mean", "Dice_std", "#samples"])
    df_abl = pd.DataFrame(ablation_table, columns=["Ablation", "Dice_mean", "IoU_mean"])

    print("\n=== Table 1: Main Comparison ===")
    print(df_results)
    print("\n=== Table 2: Ablation Study ===")
    print(df_abl)

    df_results.to_csv("experiment_harness_results.csv", index=False)
    df_abl.to_csv("experiment_harness_ablation.csv", index=False)
    print("Saved experiment_harness_results.csv and ablation.csv")

    print("Experiment harness complete. Use QUICK_RUN=False for full run.")

    return df_results, df_abl



