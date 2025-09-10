from typing import Any, Dict, List, Optional


class MetricsCollector:

    def build(
        self,
        *,
        image: Optional[Any],
        segments: Optional[List[Any]],
        zero_shot_labels: Optional[List[str]],
        predicted_classes: Optional[List[str]],
        models_used: Optional[Dict[str, Any]] = None,
        classifier_confidences: Optional[List[float]] = None,
        timings_ms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        width: Optional[int] = None
        height: Optional[int] = None
        if image is not None and hasattr(image, "size"):
            size_value = getattr(image, "size", None)
            if isinstance(size_value, (tuple, list)) and len(size_value) == 2:
                w_val, h_val = size_value
                try:
                    width = int(w_val) if w_val is not None else None
                    height = int(h_val) if h_val is not None else None
                except (TypeError, ValueError):
                    width, height = None, None

        num_segments: int = len(segments) if segments is not None else 0

        avg_segment_area: Optional[float] = None
        if segments:
            areas: List[int] = []
            for seg in segments:
                shape = getattr(seg, "shape", None)
                h: Optional[int] = None
                w: Optional[int] = None
                if isinstance(shape, (tuple, list)) and len(shape) >= 3:
                    try:
                        h = int(shape[-2])
                        w = int(shape[-1])
                    except (TypeError, ValueError):
                        h, w = None, None
                if h is None or w is None:
                    h = getattr(seg, "height", None)
                    w = getattr(seg, "width", None)
                    try:
                        h = int(h) if h is not None else None
                        w = int(w) if w is not None else None
                    except (TypeError, ValueError):
                        h, w = None, None
                if h is not None and w is not None:
                    areas.append(h * w)
            if areas:
                avg_segment_area = float(sum(areas)) / float(len(areas))

        label_source: List[str] = zero_shot_labels or predicted_classes or []
        num_object_types: int = len(set(label_source)) if label_source else 0

        type_of_object_counted: Optional[str] = None
        if label_source:
            counts: Dict[str, int] = {}
            for label in label_source:
                counts[label] = counts.get(label, 0) + 1
            type_of_object_counted = sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]

        count_of_objects: int = num_segments

        min_classifier_conf: Optional[float] = None
        if classifier_confidences:
            try:
                min_classifier_conf = float(min(classifier_confidences))
            except (TypeError, ValueError):
                min_classifier_conf = None

        timings_ms = timings_ms or {}

        return {
            "image_resolution": {
                "width": width,
                "height": height,
            },
            "type_of_object_counted": type_of_object_counted,
            "count_of_objects": count_of_objects,
            "num_segments": num_segments,
            "num_object_types": num_object_types,
            "avg_segment_resolution": avg_segment_area,
            "models_used": models_used or {},
            "classifier_confidences": classifier_confidences or [],
            "classifier_min_confidence": min_classifier_conf,
            "inference_time_ms_per_model": {
                k: float(v) for k, v in timings_ms.items()
            },
            "overall_response_ms": float(timings_ms.get("overall_ms", 0.0)) if "overall_ms" in timings_ms else None,
        }


