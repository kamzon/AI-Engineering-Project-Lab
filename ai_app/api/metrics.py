from typing import Any, Dict, Optional, List

from prometheus_client import Counter, Gauge


# Counters
PIPELINE_RUNS_TOTAL = Counter(
    "pipeline_runs_total", "Total number of pipeline runs executed"
)


# Gauges (last observed values)
IMAGE_RES_WIDTH = Gauge(
    "pipeline_image_resolution_width", "Last observed image width (pixels)"
)
IMAGE_RES_HEIGHT = Gauge(
    "pipeline_image_resolution_height", "Last observed image height (pixels)"
)

OBJECTS_COUNT = Gauge(
    "pipeline_objects_count", "Last observed count of objects determined by the application"
)
NUM_SEGMENTS = Gauge(
    "pipeline_num_segments", "Last observed number of segments found in the image"
)
NUM_OBJECT_TYPES = Gauge(
    "pipeline_num_object_types", "Last observed number of different object types found"
)
AVG_SEGMENT_RESOLUTION = Gauge(
    "pipeline_avg_segment_resolution", "Last observed average segment resolution (area in pixels)"
)

# Labelled gauges
TYPE_OF_OBJECT_COUNTED = Gauge(
    "pipeline_type_of_object_counted",
    "Indicator (1) for the type of object counted in the last run",
    ["object_type"],
)
OBJECTS_BY_TYPE = Gauge(
    "pipeline_objects_by_type",
    "Last observed number of objects per type",
    ["object_type"],
)

# Models used (as a one-hot gauge with labels carrying model identifiers)
MODELS_USED = Gauge(
    "pipeline_models_used",
    "Models used in the pipeline (value=1)",
    ["sam_variant", "sam_checkpoint", "classifier_id", "zero_shot_id"],
)

# Confidence and timing metrics
CLASSIFIER_MIN_CONFIDENCE = Gauge(
    "pipeline_classifier_min_confidence", "Minimum classifier confidence across segments"
)
CLASSIFIER_CONFIDENCE = Gauge(
    "pipeline_classifier_confidence", "Classifier confidence per segment", ["segment_index"]
)
SAM_INFERENCE_MS = Gauge(
    "pipeline_sam_inference_ms", "SAM inference time (ms) per run"
)
CLASSIFIER_INFERENCE_MS = Gauge(
    "pipeline_classifier_inference_ms", "Classifier inference time (ms) per run"
)
ZEROSHOT_INFERENCE_MS = Gauge(
    "pipeline_zeroshot_inference_ms", "Zero-shot inference time (ms) per run"
)
OVERALL_RESPONSE_MS = Gauge(
    "pipeline_overall_response_ms", "Overall pipeline response time (ms) per run"
)


def _safe_int(value: Optional[Any]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def record_pipeline_metrics(metadata: Dict[str, Any], label_counts: Dict[str, int]) -> None:
    """Record pipeline metrics to Prometheus from metadata and label counts."""
    PIPELINE_RUNS_TOTAL.inc()

    image_res = metadata.get("image_resolution", {}) or {}
    width = _safe_int(image_res.get("width"))
    height = _safe_int(image_res.get("height"))
    if width is not None:
        IMAGE_RES_WIDTH.set(width)
    if height is not None:
        IMAGE_RES_HEIGHT.set(height)

    count_of_objects = _safe_int(metadata.get("count_of_objects"))
    if count_of_objects is not None:
        OBJECTS_COUNT.set(count_of_objects)

    num_segments = _safe_int(metadata.get("num_segments"))
    if num_segments is not None:
        NUM_SEGMENTS.set(num_segments)

    num_object_types = _safe_int(metadata.get("num_object_types"))
    if num_object_types is not None:
        NUM_OBJECT_TYPES.set(num_object_types)

    avg_seg_res = metadata.get("avg_segment_resolution")
    try:
        if avg_seg_res is not None:
            AVG_SEGMENT_RESOLUTION.set(float(avg_seg_res))
    except (TypeError, ValueError):
        pass

    # Type of object counted (set indicator to 1 for the selected type)
    selected_type = metadata.get("type_of_object_counted")
    if isinstance(selected_type, str) and selected_type:
        TYPE_OF_OBJECT_COUNTED.labels(object_type=selected_type).set(1)

    # Objects by type from label_counts
    if isinstance(label_counts, dict):
        for obj_type, cnt in label_counts.items():
            try:
                OBJECTS_BY_TYPE.labels(object_type=str(obj_type)).set(float(cnt))
            except Exception:
                # Skip invalid label/count
                continue

    # Models used
    models = metadata.get("models_used") or {}
    sam = models.get("sam") or {}
    classifier = models.get("classifier") or {}
    zero_shot = models.get("zero_shot") or {}
    MODELS_USED.labels(
        sam_variant=str(sam.get("variant", "")),
        sam_checkpoint=str(sam.get("checkpoint", "")),
        classifier_id=str(classifier.get("id", "")),
        zero_shot_id=str(zero_shot.get("id", "")),
    ).set(1)

    # Classifier confidences
    confs: List[float] = [
        c for c in (metadata.get("classifier_confidences") or []) if isinstance(c, (int, float))
    ]
    if confs:
        try:
            CLASSIFIER_MIN_CONFIDENCE.set(float(min(confs)))
        except Exception:
            pass
        for idx, c in enumerate(confs):
            try:
                CLASSIFIER_CONFIDENCE.labels(segment_index=str(idx)).set(float(c))
            except Exception:
                continue

    # Timings
    timings = metadata.get("inference_time_ms_per_model") or {}
    try:
        if "sam_ms" in timings:
            SAM_INFERENCE_MS.set(float(timings["sam_ms"]))
    except Exception:
        pass
    try:
        if "classifier_ms" in timings:
            CLASSIFIER_INFERENCE_MS.set(float(timings["classifier_ms"]))
    except Exception:
        pass
    try:
        if "zero_shot_ms" in timings:
            ZEROSHOT_INFERENCE_MS.set(float(timings["zero_shot_ms"]))
    except Exception:
        pass
    try:
        overall_ms = metadata.get("overall_response_ms")
        if overall_ms is not None:
            OVERALL_RESPONSE_MS.set(float(overall_ms))
    except Exception:
        pass


