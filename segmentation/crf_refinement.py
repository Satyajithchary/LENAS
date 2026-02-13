def apply_random_walker_refinement(image_np, mask_np, beta=10, mode='bf'):

    if not RW_AVAILABLE:
        return mask_np
    try:
        img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # markers: 0 unknown, 1 background, 2 foreground
        markers = np.zeros(mask_np.shape, dtype=np.int32)
        markers[mask_np==1] = 2
        # background: erode inverse
        markers[(mask_np==0) & (cv2.erode((mask_np==0).astype(np.uint8), np.ones((3,3),np.uint8), iterations=3)==1)] = 1
        rw = random_walker(img_gray.astype(np.float32), markers, beta=beta, mode=mode)
        refined = (rw==2).astype(np.uint8)
        return refined
    except Exception as e:
        print(f"random_walker refinement failed: {e}")
        return mask_np
