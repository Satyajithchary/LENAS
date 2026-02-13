def evaluate_otsu(dataset, num_samples=None):
    """
    dataset: dataset object returning (image_tensor, mask_tensor, path)
    returns: list of (dice, iou)
    """
    results = []
    n = len(dataset)
    indices = np.arange(n)
    if num_samples is not None:
        indices = np.random.choice(indices, min(num_samples, n), replace=False)

    for idx in indices:
        img_tensor, mask_tensor, path = dataset[idx]
        img = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred = (th/255).astype(np.uint8)
        gt = tensor_to_numpy_mask(mask_tensor)
        dice = calculate_dice_score(pred, gt)
        iou = calculate_iou_score(pred, gt)
        results.append((dice, iou))
    return results
