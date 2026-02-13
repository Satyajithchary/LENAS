
class XAICache:
    """Cache for XAI computations"""
    def __init__(self):
        self.cache = {}

    def get_key(self, image_tensor, label):
        # Simple hash-based key
        img_hash = hash(image_tensor.cpu().numpy().tobytes())
        return f"{img_hash}_{label}"

    def get(self, image_tensor, label):
        key = self.get_key(image_tensor, label)
        return self.cache.get(key, None)

    def set(self, image_tensor, label, explanations):
        key = self.get_key(image_tensor, label)
        self.cache[key] = explanations

    def clear(self):
        self.cache.clear()

# Global XAI cache
xai_cache = XAICache()