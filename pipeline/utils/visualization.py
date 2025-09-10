from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class PanopticVisualizer:
    def __init__(self) -> None:
        pass

    def save(self, panoptic_map, output_path: str, labels: Optional[List[str]] = None, annotate: bool = True) -> str:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        panoptic_np = panoptic_map.cpu().numpy().astype(np.int32)
        max_label = int(panoptic_np.max()) if panoptic_np.size > 0 else 0
        if max_label == 0:
            color_img = np.zeros((*panoptic_np.shape, 3), dtype=np.uint8)
        else:
            cmap = plt.get_cmap("tab20", max(20, max_label + 1))
            colors = (cmap(np.arange(max_label + 1))[:, :3] * 255).astype(np.uint8)
            color_img = colors[panoptic_np]
            color_img[panoptic_np == 0] = 0
        img_pil = Image.fromarray(color_img, mode="RGB")

        if annotate:
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            labels_present = [int(l) for l in np.unique(panoptic_np).tolist() if int(l) != 0]
            for label_id in labels_present:
                mask_np = (panoptic_np == label_id)
                ys, xs = np.where(mask_np)
                if ys.size == 0:
                    continue
                cx = int(xs.mean())
                cy = int(ys.mean())
                text = str(label_id)
                if labels and 0 <= (label_id - 1) < len(labels):
                    text = f"{label_id}: {labels[label_id - 1]}"
                try:
                    draw.text((cx, cy), text, fill=(255, 255, 255), font=font,
                              stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")
                except TypeError:
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except Exception:
                        tw, th = (8 * len(text), 12)
                    tx = int(cx - tw / 2)
                    ty = int(cy - th / 2)
                    bg_pad = 2
                    try:
                        draw.rectangle(
                            [tx - bg_pad, ty - bg_pad, tx + tw + bg_pad, ty + th + bg_pad], fill=(0, 0, 0))
                    except Exception:
                        pass
                    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

        img_pil.save(output_path)
        return output_path


