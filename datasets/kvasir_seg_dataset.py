
class KvasirSEGDataset(Dataset):
    """Dataset for Kvasir-SEG Polyp Segmentation"""
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Get all image and mask paths
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                 if f.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                                if f.endswith('.jpg')])

        # Verify matching
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"

        # Verify filenames match
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            assert img_name == mask_name, f"Filename mismatch: {img_name} vs {mask_name}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # Load image and mask
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale

            # Convert mask to binary (0 or 1)
            mask_array = np.array(mask)
            mask_array = (mask_array > 127).astype(np.float32)  # Threshold at 127
            mask = Image.fromarray(mask_array)

            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)

            if self.mask_transform:
                mask_tensor = self.mask_transform(mask)
            else:
                # Default mask transformation
                mask_tensor = transforms.ToTensor()(mask)

            return image_tensor, mask_tensor, img_path

        except Exception as e:
            print(f"Error loading sample {img_path}: {e}")
            return None, None, None

def collate_fn_kvasir(batch):
    # Filter out samples that failed to load
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0), []

    images, masks, paths = zip(*batch)
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)

    return images, masks, paths
