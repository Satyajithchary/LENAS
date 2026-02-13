
def train_unet_partial(model, train_loader, val_loader, device, fraction=0.1, epochs=3, force_retrain=False):
    """Train UNet model using only a fraction of available labels."""
    ckpt_path = f"checkpoints/unet_{int(fraction*100)}pct.pth"
    os.makedirs("checkpoints", exist_ok=True)

    if not force_retrain and os.path.exists(ckpt_path):
        print(f"📦 Loading cached UNet ({int(fraction*100)}%) checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # select subset of data for partial supervision
    total_batches = len(train_loader)
    subset_batches = max(1, int(total_batches * fraction))

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (imgs, masks, _) in enumerate(train_loader):
            if i >= subset_batches:
                break
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Fraction {fraction*100:.0f}% | Loss: {running_loss/subset_batches:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved UNet ({int(fraction*100)}%) checkpoint at {ckpt_path}")
    return model


def train_unet_fully_supervised(unet, train_loader, val_loader, device, epochs=5, lr=1e-4, save_path=None):
    unet.to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val = 0.0
    for ep in range(epochs):
        unet.train()
        for imgs, masks, paths in train_loader:
            if imgs.nelement() == 0: continue
            imgs = imgs.to(device); masks = masks.to(device)
            logits = unet(imgs)
            loss = criterion(logits, masks)
            opt.zero_grad(); loss.backward(); opt.step()
        val_dices = []
        with torch.no_grad():
            unet.eval()
            for imgs, masks, paths in val_loader:
                if imgs.nelement() == 0: continue
                imgs = imgs.to(device)
                probs = torch.sigmoid(unet(imgs)).cpu().numpy()
                for p,g in zip(probs, masks.cpu().numpy()):
                    dice = calculate_dice_score((p.squeeze()>0.5).astype(np.uint8), g.squeeze())
                    val_dices.append(dice)
        val_mean = np.mean(val_dices) if val_dices else 0.0
        if save_path and val_mean > best_val:
            torch.save(unet.state_dict(), save_path)
            best_val = val_mean
    return unet
