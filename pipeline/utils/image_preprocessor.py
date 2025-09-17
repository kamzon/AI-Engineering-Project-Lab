from typing import Callable, Optional

from PIL import Image, ImageFilter
import numpy as np
import random
from pipeline.config import ModelConstants

class ImagePreprocessor:
    def __init__(self, target_size: int = ModelConstants.IMAGE_RESIZE_SIZE) -> None:
        self.target_size = target_size

    def to_rgb_and_resize(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if image.size != (self.target_size, self.target_size):
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = getattr(Image, "LANCZOS", None) or getattr(Image, "BICUBIC", 0)
            image = image.resize((self.target_size, self.target_size), resample=resample)
        return image

    def build_augment(self, blur_max: float = 2.0, rotate_choices: Optional[list] = None, noise_max: float = 10.0) -> Callable[[Image.Image], Image.Image]:
        rotate_choices = rotate_choices or [0, 90, 180, 270]

        def _augment(img: Image.Image) -> Image.Image:
            img_out = img
            # Random blur
            blur_radius = random.uniform(0.0, blur_max)
            if blur_radius > 0:
                img_out = img_out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Random rotate
            rotate_angle = random.choice(rotate_choices)
            if rotate_angle != 0:
                img_out = img_out.rotate(rotate_angle)
            # Random Gaussian noise
            noise_std = random.uniform(0.0, noise_max)
            if noise_std > 0:
                arr = np.array(img_out)
                noise_arr = np.random.normal(0, noise_std, arr.shape)
                arr = np.clip(arr.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)
                img_out = Image.fromarray(arr)
            return img_out

        return _augment
