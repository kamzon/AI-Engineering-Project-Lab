from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image


class ImageUtils:
    """
    Centralized image processing utilities used across the pipeline.

    All methods are stateless and safe to use anywhere in the codebase.
    """

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    @staticmethod
    def _get_lanczos_resample() -> int:
        try:
            return Image.Resampling.LANCZOS  # Pillow >= 10
        except AttributeError:
            return getattr(Image, "LANCZOS", None) or getattr(Image, "BICUBIC", 0)

    @staticmethod
    def resize_square(image: Image.Image, size: int) -> Image.Image:
        if image.size == (size, size):
            return image
        resample = ImageUtils._get_lanczos_resample()
        return image.resize((size, size), resample=resample)

    @staticmethod
    def resize_longest_side(image: Image.Image, target_longest_side: int) -> Image.Image:
        width, height = image.size
        longest_side = max(width, height)
        if longest_side == target_longest_side:
            return image
        scale = target_longest_side / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample = ImageUtils._get_lanczos_resample()
        return image.resize((new_width, new_height), resample=resample)

    @staticmethod
    def pil_to_tensor_uint8_chw(image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to torch uint8 tensor in CHW layout.
        """
        transform = tv_transforms.Compose([tv_transforms.PILToTensor()])
        tensor = transform(image)
        # Ensure dtype is uint8
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        return tensor

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """
        Convert a torch tensor in CHW or HWC layout (uint8 or float [0,1]) to PIL RGB image.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("tensor_to_pil expects a torch.Tensor")
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            chw = tensor
        elif tensor.ndim == 3 and tensor.shape[2] in (1, 3):
            chw = tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                "Unsupported tensor shape for image; expected CHW or HWC with 1 or 3 channels"
            )
        if chw.dtype.is_floating_point:
            if chw.max() <= 1.0:
                chw = (chw * 255.0).round()
        chw = chw.clamp(0, 255).byte()
        np_img = chw.permute(1, 2, 0).numpy()
        if np_img.shape[2] == 1:
            return Image.fromarray(np_img[:, :, 0], mode="L").convert("RGB")
        return Image.fromarray(np_img, mode="RGB")

    @staticmethod
    def ensure_pil_image(image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            return ImageUtils.tensor_to_pil(image)
        raise TypeError("image must be a PIL.Image.Image or torch.Tensor")

    @staticmethod
    def load_image(path: str, target_longest_side: Optional[int] = None) -> Image.Image:
        """
        Load an image from disk, convert to RGB, and optionally resize by longest side.
        """
        img = Image.open(path)
        img = ImageUtils.to_rgb(img)
        if target_longest_side is not None:
            img = ImageUtils.resize_longest_side(img, target_longest_side)
        return img

    @staticmethod
    def build_resize_to_tensor(size_hw: Tuple[int, int]) -> Callable[[Image.Image], torch.Tensor]:
        """
        Returns a transform that resizes an image to size_hw (H, W) and converts to float tensor in [0,1].
        """
        return tv_transforms.Compose([
            tv_transforms.Resize(size_hw),
            tv_transforms.ToTensor(),
        ])

    @staticmethod
    def crop_by_bbox(
        image: Image.Image,
        bbox: List[float],
        min_size: int = 32,
        background_fill: int = 188,
    ) -> torch.Tensor:
        """
        Crop image by bbox [x1, y1, x2, y2] and return CHW uint8 tensor, padded to at least min_size.
        """
        img_tensor = ImageUtils.pil_to_tensor_uint8_chw(image)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return torch.full((3, min_size, min_size), background_fill, dtype=torch.uint8)

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_height, img_width = img_tensor.shape[1], img_tensor.shape[2]
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))
        cropped = img_tensor[:, y1:y2, x1:x2]
        if cropped.shape[1] < min_size or cropped.shape[2] < min_size:
            pad_h = max(0, min_size - cropped.shape[1])
            pad_w = max(0, min_size - cropped.shape[2])
            cropped = torch.nn.functional.pad(
                cropped.float(),
                (0, pad_w, 0, pad_h),
                value=background_fill,
            ).to(torch.uint8)
        return cropped

    @staticmethod
    def crop_segments_from_detections(
        image: Image.Image,
        detections: List[Dict[str, Union[int, float, str, List[float]]]],
        background_fill: int = 188,
        min_size: int = 32,
    ) -> List[torch.Tensor]:
        segments: List[torch.Tensor] = []
        for det in detections:
            bbox = det.get("bbox", []) if isinstance(det, dict) else []
            seg = ImageUtils.crop_by_bbox(image, bbox, min_size=min_size, background_fill=background_fill)
            segments.append(seg)
        return segments


