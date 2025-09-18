from django.test import SimpleTestCase
from .metrics import record_pipeline_metrics


class MetricsRecordingTests(SimpleTestCase):
    def test_record_pipeline_metrics_full_metadata(self):
        metadata = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "image_resolution": {"width": 640, "height": 480},
            "count_of_objects": 3,
            "num_segments": 3,
            "num_object_types": 1,
            "avg_segment_resolution": 1234.5,
            "type_of_object_counted": "cat",
            "models_used": {
                "sam": {"variant": "sam2", "checkpoint": "sam2_huge.pth"},
                "classifier": {"id": "microsoft/resnet-50"},
                "zero_shot": {"id": "facebook/bart-large-mnli"},
            },
            "classifier_confidences": [0.9, 0.8, 0.7],
            "inference_time_ms_per_model": {
                "safety_ms": 1.0,
                "sam_ms": 2.0,
                "classifier_ms": 3.0,
                "zero_shot_ms": 4.0,
            },
            "overall_response_ms": 10.0,
            "safety": {"label": "safe", "confidence": 0.95},
        }
        label_counts = {"cat": 3, "other": 0}
        # Should not raise
        record_pipeline_metrics(metadata, label_counts)

    def test_record_pipeline_metrics_partial_metadata(self):
        # Exercise exception handling paths by omitting most fields
        metadata = {}
        label_counts = {}
        record_pipeline_metrics(metadata, label_counts)


