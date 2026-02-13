def generate_scribbles_from_mask(mask_tensor):
    """
    Converts binary mask tensors to sparse scribble supervision.
    """
    scribbles = []
    for mask in mask_tensor:
        mask_np = mask.squeeze().detach().cpu().numpy()
        mask_np = (mask_np > 0).astype(np.uint8)

        coords = np.column_stack(np.where(mask_np > 0))
        np.random.shuffle(coords)
        sample_coords = coords[:max(1, len(coords)//50)]  # sparse points
        scribble_mask = np.zeros_like(mask_np)
        for y, x in sample_coords:
            scribble_mask[y, x] = 1

        scribbles.append(torch.tensor(scribble_mask, dtype=torch.float32).unsqueeze(0))

    return torch.stack(scribbles).to(mask_tensor.device)  # ensure CUDA match


def train_unet_with_scribbles(model, train_loader, val_loader, device, epochs=3,
                              scribble_getter=None, force_retrain=False,
                              checkpoint_path="scribble_unet.pth"):

    # If checkpoint exists and not forced to retrain
    if not force_retrain and os.path.exists(checkpoint_path):
        print(f"🟡 Loading cached ScribbleUNet weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    print(f"Training ScribbleUNet for {epochs} epochs on {device}...")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Generate scribbles (if provided)
            if scribble_getter is not None:
                scribbles = scribble_getter(masks)
                scribbles = scribbles.to(device)  
                masks = scribbles  

            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Dice: {avg_dice:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved ScribbleUNet weights to {checkpoint_path}")
    return model
