
def validate_classifier_with_uncertainty(model, dataloader, criterion, device, uncertainty_threshold=None):
    """Validation with uncertainty detection."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_entropies = []
    uncertain_count = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validating", leave=False):
            if images.nelement() == 0: continue # Skip empty batches

            images, labels = images.to(device), labels.to(device)
            predictions, entropy, probs = model.predict_with_uncertainty(images)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()
            all_entropies.extend(entropy.cpu().numpy())

            if uncertainty_threshold is not None:
                uncertain_count += (entropy > uncertainty_threshold).sum().item()

    avg_entropy = np.mean(all_entropies) if all_entropies else 0
    val_acc = (val_correct / val_total * 100) if val_total > 0 else 0
    val_loss = (val_loss / val_total) if val_total > 0 else 0

    if uncertainty_threshold is not None:
        print(f"Uncertain predictions: {uncertain_count}/{val_total}")

    return val_loss, val_acc, avg_entropy

