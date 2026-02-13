
class CapsuleEndoscopyDataset(Dataset):
    """Dataset for Capsule Endoscopy Classification (10 classes)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        if not os.path.isdir(root_dir):
            return

        self.class_to_idx = {d: i for i, d in enumerate(sorted(os.listdir(root_dir)))}
        self.idx_to_class = {i: d for d, i in self.class_to_idx.items()}

        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for subfolder in os.listdir(class_dir):
                    subfolder_path = os.path.join(class_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        for img_name in os.listdir(subfolder_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(subfolder_path, img_name),
                                                   self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except:
            return None, None, None

def collate_fn_capsule(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.empty(0), torch.empty(0), [])
