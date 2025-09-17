from typing import List, Dict, Any
import torch
import torchvision.transforms as tf
from PIL import Image


class SegmentCropper:
    """Utility to crop image segments from detection bounding boxes."""
    
    def __init__(self, background_fill: int = 188):
        self.background_fill = background_fill
    
    def crop_segments_from_detections(
        self, 
        image: Image.Image, 
        detections: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """
        Crop image segments from detection bounding boxes.
        
        Args:
            image: PIL Image
            detections: List of detection dicts with 'bbox' key containing [x1, y1, x2, y2]
        
        Returns:
            List of cropped segments as torch tensors in CHW format
        """
        segments = []
        
        # Convert PIL image to tensor for processing
        transform = tf.Compose([tf.PILToTensor()])
        image_tensor = transform(image)  # Shape: (C, H, W)
        
        for detection in detections:
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                # Create a dummy segment filled with background color if bbox is invalid
                dummy_segment = torch.full((3, 64, 64), self.background_fill, dtype=torch.uint8)
                segments.append(dummy_segment)
                continue
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))
            
            # Crop the segment
            cropped = image_tensor[:, y1:y2, x1:x2]
            
            # Ensure minimum size (ResNet needs reasonable input size)
            if cropped.shape[1] < 32 or cropped.shape[2] < 32:
                # Pad small segments to minimum size
                pad_h = max(0, 32 - cropped.shape[1])
                pad_w = max(0, 32 - cropped.shape[2])
                cropped = torch.nn.functional.pad(
                    cropped.float(), 
                    (0, pad_w, 0, pad_h), 
                    value=self.background_fill
                ).to(torch.uint8)
            
            segments.append(cropped)
        
        return segments
