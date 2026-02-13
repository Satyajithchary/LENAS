
from models.differential_biomedclip import DifferentialBiomedCLIP
from datasets.capsule_dataset import CapsuleEndoscopyDataset
from training.train_classifier import train_classifier_with_hybrid_loss
from xai.generate_explanations import generate_explanations_focused
from xai.prompt_selection import uncertainty_guided_prompts

def main():
    """Main execution pipeline for both capsule endoscopy and Kvasir-SEG."""
    clear_output()

    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    SAM_CKPT_PATH = "./sam_vit_b_01ec64.pth"

    # Paths for both datasets
    CAPSULE_DATA_PATH = "/path/to/capsule/training"
    KVASIR_SEG_DATA_DIR = "/path/to/Kvasir-Seg"

    # Capsule endoscopy configuration
    CAPSULE_NUM_CLASSES = 10
    CAPSULE_BATCH_SIZE = 16
    CAPSULE_EPOCHS = 1
    CAPSULE_LR = 1e-5
    CAPSULE_CLASS_LABELS = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
                           'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

    CAPSULE_MODEL_SAVE_PATH = "best_capsule_endoscopy_model.pth"

    print(f"Using device: {DEVICE}")
    print(f"SAM Available: {SAM_AVAILABLE}")

    # ========== STAGE 1: Train on Capsule Endoscopy Dataset ==========
    print("\n" + "="*80)
    print("STAGE 1: Training on Capsule Endoscopy Dataset (10 classes)")
    print("="*80)

    capsule_model = DifferentialBiomedCLIP(MODEL_NAME, CAPSULE_NUM_CLASSES, DEVICE,
                                          class_names=CAPSULE_CLASS_LABELS, use_contrastive=True)

    # Load capsule endoscopy dataset
    capsule_dataset = CapsuleEndoscopyDataset(
        root_dir=CAPSULE_DATA_PATH,
        transform=capsule_model.preprocess
    )

    if not capsule_dataset.samples:
        print(f"❌ No samples found in capsule dataset at {CAPSULE_DATA_PATH}")
        return None, None, None

    print(f"Total capsule endoscopy samples: {len(capsule_dataset)}")

    # Split dataset
    train_size = int(0.8 * len(capsule_dataset))
    val_size = len(capsule_dataset) - train_size
    train_ds, val_ds = random_split(capsule_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, CAPSULE_BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn_capsule, num_workers=2)
    val_loader = DataLoader(val_ds, CAPSULE_BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn_capsule, num_workers=2)

    # Train or load model
    if os.path.exists(CAPSULE_MODEL_SAVE_PATH):
        print(f"Loading existing capsule model from {CAPSULE_MODEL_SAVE_PATH}")
        try:
            checkpoint = torch.load(CAPSULE_MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
            capsule_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded capsule model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.2f}%")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training capsule model from scratch...")
            history = train_classifier_with_hybrid_loss(
                capsule_model, train_loader, val_loader, CAPSULE_EPOCHS, CAPSULE_LR, DEVICE,
                CAPSULE_CLASS_LABELS, CAPSULE_MODEL_SAVE_PATH,
                contrastive_weight=0.3, classification_weight=0.7
            )
    else:
        print("No existing capsule model found. Training from scratch...")
        history = train_classifier_with_hybrid_loss(
            capsule_model, train_loader, val_loader, CAPSULE_EPOCHS, CAPSULE_LR, DEVICE,
            CAPSULE_CLASS_LABELS, CAPSULE_MODEL_SAVE_PATH,
            contrastive_weight=0.3, classification_weight=0.7
        )

    # ========== STAGE 2: XAI Evaluation on Capsule Model ==========
    print("\n" + "="*80)
    print("STAGE 2: XAI Evaluation on Capsule Model")
    print("="*80)

    # Use pre-computed weights or evaluate
    use_cached_weights = False
    if use_cached_weights:
        xai_weights = {'Saliency': 0.279, 'IntegratedGradients': 0.352, 'GradientShap': 0.369}
        print(f"Using pre-computed XAI weights: {xai_weights}")
    else:
        xai_weights, xai_metrics = evaluate_xai_methods_quantus(
            capsule_model, val_loader, DEVICE, CAPSULE_NUM_CLASSES,
            num_samples=20, CLASS_LABELS=CAPSULE_CLASS_LABELS
        )

    # ========== STAGE 3: Setup SAM for Segmentation ==========
    print("\n" + "="*80)
    print("STAGE 3: Setting up SAM for Kvasir-SEG Segmentation")
    print("="*80)

    sam_predictor = None
    if SAM_AVAILABLE and os.path.exists(SAM_CKPT_PATH):
        try:
            sam_model = sam_model_registry['vit_b'](checkpoint=SAM_CKPT_PATH)
            sam_model = sam_model.to(DEVICE)
            sam_predictor = SamPredictor(sam_model)
            print("✅ SAM model loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM: {e}")
            print("Will use fallback segmentation.")
    else:
        print(f"SAM checkpoint not found at {SAM_CKPT_PATH}")
        print("Will use fallback segmentation.")

    # ========== STAGE 4: Kvasir-SEG Segmentation Evaluation ==========
    print("\n" + "="*80)
    print("STAGE 4: Kvasir-SEG Segmentation Evaluation")
    print("="*80)

    # Load Kvasir-SEG dataset
    kvasir_image_dir = os.path.join(KVASIR_SEG_DATA_DIR, 'images')
    kvasir_mask_dir = os.path.join(KVASIR_SEG_DATA_DIR, 'masks')

    if not os.path.exists(kvasir_image_dir) or not os.path.exists(kvasir_mask_dir):
        print(f"❌ Kvasir-SEG dataset not found at {KVASIR_SEG_DATA_DIR}")
        print("Please update the KVASIR_SEG_DATA_DIR path")
        return capsule_model, sam_predictor, xai_weights

    # Create transforms for Kvasir-SEG
    kvasir_transform = capsule_model.preprocess  # Use same transform as training
    kvasir_mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    kvasir_dataset = KvasirSEGDataset(
        image_dir=kvasir_image_dir,
        mask_dir=kvasir_mask_dir,
        transform=kvasir_transform,
        mask_transform=kvasir_mask_transform
    )

    print(f"Loaded {len(kvasir_dataset)} Kvasir-SEG samples")

    # First, let's visualize some samples from Kvasir-SEG
    print("\n📊 Visualizing Kvasir-SEG samples...")
    num_sample_viz = min(5, len(kvasir_dataset))
    sample_indices = np.random.choice(len(kvasir_dataset), num_sample_viz, replace=False)

    fig, axes = plt.subplots(num_sample_viz, 3, figsize=(15, 5*num_sample_viz))
    if num_sample_viz == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        image_tensor, mask_tensor, img_path = kvasir_dataset[idx]

        # Convert back to numpy for visualization
        image_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = mask_tensor.squeeze().cpu().numpy()

        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original: {os.path.basename(img_path)}")
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(image_np)
        axes[i, 2].imshow(mask_np, alpha=0.5, cmap='jet')
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.suptitle("Kvasir-SEG Sample Visualizations", fontsize=16, fontweight='bold')
    plt.show()

    # Now run segmentation on Kvasir-SEG dataset
    print("\n🎯 Running segmentation on Kvasir-SEG dataset...")

    kvasir_loader = DataLoader(kvasir_dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn_kvasir)

    all_dice_scores = []
    all_iou_scores = []
    segmentation_results = []

    # Process each sample
    for i, (images, masks, paths) in tqdm(enumerate(kvasir_loader), total=len(kvasir_loader), desc="Segmenting Kvasir-SEG"):
        if images.nelement() == 0:
            continue

        image_tensor = images.squeeze(0)
        mask_tensor = masks.squeeze(0)
        img_path = paths[0]

        sample = (image_tensor, mask_tensor, img_path)

        # Run segmentation pipeline with visualization
        try:
            pred_mask, dice, iou, iter_history = visualize_kvasir_segmentation_pipeline(
                sample, capsule_model, sam_predictor, DEVICE, xai_weights
            )

            segmentation_results.append({
                'image_path': img_path,
                'dice': dice,
                'iou': iou,
                'pred_mask': pred_mask,
                'iter_history': iter_history
            })

            all_dice_scores.append(dice)
            all_iou_scores.append(iou)

            print(f"Sample {i+1}/{len(kvasir_loader)} - Dice: {dice:.4f}, IoU: {iou:.4f}")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue

        # Clear cache every 10 samples to prevent memory issues
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ========== STAGE 5: Final Results and Summary ==========
    print("\n" + "="*80)
    print("STAGE 5: Final Results Summary")
    print("="*80)

    if all_dice_scores:
        avg_dice = np.mean(all_dice_scores)
        avg_iou = np.mean(all_iou_scores)
        std_dice = np.std(all_dice_scores)
        std_iou = np.std(all_iou_scores)

        print(f"\n📊 Kvasir-SEG Segmentation Results:")
        print(f"   Total Samples Processed: {len(all_dice_scores)}")
        print(f"   Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"   Average IoU Score:  {avg_iou:.4f} ± {std_iou:.4f}")
        print(f"   Best Dice Score:    {np.max(all_dice_scores):.4f}")
        print(f"   Best IoU Score:     {np.max(all_iou_scores):.4f}")
        print(f"   Worst Dice Score:   {np.min(all_dice_scores):.4f}")
        print(f"   Worst IoU Score:    {np.min(all_iou_scores):.4f}")

        # Plot distribution of scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.hist(all_dice_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(avg_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dice:.4f}')
        ax1.set_xlabel('Dice Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Dice Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(all_iou_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(avg_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_iou:.4f}')
        ax2.set_xlabel('IoU Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of IoU Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Save results to CSV
        results_df = pd.DataFrame(segmentation_results)
        results_df.to_csv('kvasir_segmentation_results.csv', index=False)
        print(f"\n💾 Results saved to 'kvasir_segmentation_results.csv'")

    else:
        print("❌ No successful segmentations to report.")

    # Clear XAI cache
    xai_cache.clear()

    print("\n" + "="*80)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("="*80)

    print(f"\n📝 Pipeline Summary:")
    print(f"   • Capsule Endoscopy Model: Trained on {len(capsule_dataset)} samples")
    print(f"   • Kvasir-SEG Segmentation: Processed {len(all_dice_scores) if all_dice_scores else 0} samples")
    print(f"   • XAI Methods: {len(xai_weights)} methods with optimized weights")
    print(f"   • SAM Integration: {'Enabled' if sam_predictor else 'Disabled'}")

    return capsule_model, sam_predictor, xai_weights

# =========================================================================================
# EXECUTE MAIN PIPELINE
# =========================================================================================

if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE MEDICAL IMAGE ANALYSIS PIPELINE")
    print("Capsule Endoscopy Classification + Kvasir-SEG Segmentation")
    print("="*80)

    try:
        result = main()

        if result is not None:
            capsule_model, sam_predictor, xai_weights = result
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline returned None - check for errors above")

    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("🎉 CODE EXECUTION FINISHED!")
print("="*80)