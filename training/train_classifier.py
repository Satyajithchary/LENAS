
def train_classifier_with_hybrid_loss(model, train_loader, val_loader, epochs, lr, device,
                                     CLASS_LABELS, save_path="best_differential_biomedclip.pth",
                                     contrastive_weight=0.3, classification_weight=0.7):
    """
    Training with hybrid contrastive + classification loss
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_contrastive = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_loss_contrastive': [], 'train_loss_cls': [],
               'val_loss': [], 'val_acc': [], 'val_entropy': []}

    for epoch in range(epochs):
        model.train()
        running_loss, running_loss_contrastive, running_loss_cls = 0.0, 0.0, 0.0
        correct_predictions, total_samples = 0, 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            if images.nelement() == 0: continue # Skip empty batches

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Get both contrastive and classification logits
            if model.use_contrastive:
                logits_contrastive, logits_cls = model(images, return_contrastive=True)

                # Hybrid loss
                loss_contrastive = criterion_contrastive(logits_contrastive, labels)
                loss_cls = criterion_cls(logits_cls, labels)
                loss = contrastive_weight * loss_contrastive + classification_weight * loss_cls

                running_loss_contrastive += loss_contrastive.item() * images.size(0)
                running_loss_cls += loss_cls.item() * images.size(0)
            else:
                logits_cls = model(images)
                loss = criterion_cls(logits_cls, labels)
                running_loss_cls += loss.item() * images.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, pred = torch.max(logits_cls.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (pred == labels).sum().item()

        avg_train_loss = running_loss / total_samples if total_samples > 0 else 0
        avg_train_loss_contrastive = running_loss_contrastive / total_samples if model.use_contrastive and total_samples > 0 else 0
        avg_train_loss_cls = running_loss_cls / total_samples if total_samples > 0 else 0

        val_loss, val_acc, val_entropy = validate_classifier_with_uncertainty(
            model, val_loader, criterion_cls, device)

        history['train_loss'].append(avg_train_loss)
        history['train_loss_contrastive'].append(avg_train_loss_contrastive)
        history['train_loss_cls'].append(avg_train_loss_cls)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_entropy'].append(val_entropy)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} " +
              (f"(Contr: {avg_train_loss_contrastive:.4f}, Cls: {avg_train_loss_cls:.4f}), " if model.use_contrastive else "") +
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Avg Entropy: {val_entropy:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            print(f"New best validation accuracy! Saving to {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            best_val_acc = val_acc

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.2f}%")
    return history