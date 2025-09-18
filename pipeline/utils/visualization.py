from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class PanopticVisualizer:
    def __init__(self) -> None:
        pass

    def save_detections(self, image: Image.Image, detections: List[dict], output_path: str) -> str:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in detections:
            bbox = det.get("bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(x) for x in bbox]
            # Prefer ResNet classification label/confidence if present
            preferred_label = det.get("resnet_label") or det.get("label") or "object"
            label = str(preferred_label)
            score = det.get("resnet_conf")
            if score is None:
                score = det.get("score")
            outline = (0, 255, 0)
            try:
                draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
            except Exception:
                draw.rectangle([x1, y1, x2, y2], outline=outline)
            text = f"{label}"
            if isinstance(score, (int, float)):
                text += f" {score:.2f}"
            # draw filled text background
            try:
                bbox_txt = draw.textbbox((0, 0), text, font=font)
                tw = bbox_txt[2] - bbox_txt[0]
                th = bbox_txt[3] - bbox_txt[1]
            except Exception:
                tw, th = (8 * len(text), 12)
            pad = 2
            tx = int(x1)
            ty = max(0, int(y1) - th - 2 * pad)
            try:
                draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
            except Exception:
                pass
            draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

        img.save(output_path)
        return output_path


