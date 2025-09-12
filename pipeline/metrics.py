from typing import Any, Dict, List, Optional

class MetricsCollector:

    def _compute_classification_metrics(self) -> Dict[str, Optional[float]]:
        """
        Compute dataset-level accuracy, precision, and recall from DB records.

        Accuracy rule:
        - If corrected_count is present and != predicted_count → miss
        - If corrected_count is present and == predicted_count → hit
        - If corrected_count is None → assume hit

        Precision/Recall count-level approximation (only on records with corrections):
        - TP += min(predicted_count, corrected_count)
        - FP += max(predicted_count - corrected_count, 0)
        - FN += max(corrected_count - predicted_count, 0)
        """
        # Avoid touching Django models before apps are ready (e.g., during checks/tests)
        try:
            from django.apps import apps as django_apps  # type: ignore  # pylint: disable=import-outside-toplevel
            if not getattr(django_apps, "ready", False):
                return {"accuracy": None, "precision": None, "recall": None}
        except Exception:
            return {"accuracy": None, "precision": None, "recall": None}
        try:
            from records.models import Result  # type: ignore  # pylint: disable=import-outside-toplevel
        except Exception:
            print("Error importing Result model")
            return {"accuracy": None, "precision": None, "recall": None}

        try:
            queryset = Result.objects.all()
            total: int = 0
            hits: int = 0

            tp: int = 0
            fp: int = 0
            fn: int = 0

            for r in queryset:
                predicted_raw = getattr(r, "predicted_count", None)
                corrected_raw = getattr(r, "corrected_count", None)

                predicted = int(predicted_raw) if predicted_raw is not None else 0
                corrected = int(corrected_raw) if corrected_raw is not None else None

                # Accuracy tally (assume hit if no correction)
                total += 1
                if corrected is None:
                    hits += 1
                else:
                    hits += 1 if predicted == corrected else 0

                # Precision/Recall only when corrected is available
                if corrected is not None:
                    tp += min(predicted, corrected)
                    if predicted > corrected:
                        fp += (predicted - corrected)
                    elif corrected > predicted:
                        fn += (corrected - predicted)

            accuracy: Optional[float] = None if total == 0 else float(hits) / float(total)
            precision: Optional[float] = None
            recall: Optional[float] = None
            if (tp + fp) > 0:
                precision = float(tp) / float(tp + fp)
            if (tp + fn) > 0:
                recall = float(tp) / float(tp + fn)

            return {"accuracy": accuracy, "precision": precision, "recall": recall}
        except Exception:
            return {"accuracy": None, "precision": None, "recall": None}

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

        pr_metrics = self._compute_classification_metrics()

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
            "accuracy": pr_metrics.get("accuracy"),
            "precision": pr_metrics.get("precision"),
            "recall": pr_metrics.get("recall"),
            "inference_time_ms_per_model": {
                k: float(v) for k, v in timings_ms.items()
            },
            "overall_response_ms": float(timings_ms.get("overall_ms", 0.0)) if "overall_ms" in timings_ms else None,
        }


