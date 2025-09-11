# API Reference

Base URL: `http://127.0.0.1:8000`

Auth: None (local dev)

---

## POST /api/count/
Upload an image and the target object type to count.

- Content-Type: multipart/form-data
- Form fields:
  - `image`: file (png/jpg)
  - `object_type`: string (e.g., `cat`)

Response 200 (JSON, example):
```json
{
  "id": 42,
  "object_type": "cat",
  "counts": {"cat": 2, "dog": 0},
  "num_segments": 3,
  "panoptic_path": "pipeline/outputs/abcdef123.png",
  "metadata": {
    "image_resolution": {"width": 1024, "height": 768},
    "inference_time_ms_per_model": {
      "sam_ms": 120.5,
      "classifier_ms": 60.2,
      "zero_shot_ms": 15.0,
      "overall_ms": 220.9
    }
  }
}
```

Errors:
- 400 if fields are missing or invalid
- 500 on internal errors

Example curl:
```bash
curl -X POST http://127.0.0.1:8000/api/count/ \
  -F "object_type=cat" \
  -F "image=@/path/to/file.png"
```

---

## POST /api/correct/
Submit a corrected count for a previous result.

- Content-Type: application/x-www-form-urlencoded or JSON
- Fields:
  - `result_id`: integer (the id returned by /api/count/)
  - `corrected_count`: integer

Response 200 (JSON, example):
```json
{"status": "ok", "result_id": 42, "corrected_count": 3}
```

Errors:
- 400 if fields are missing/invalid
- 404 if result not found

Example curl:
```bash
curl -X POST http://127.0.0.1:8000/api/correct/ \
  -d "result_id=42" -d "corrected_count=3"
```

---

## GET /prometheus/metrics
Prometheus exposition endpoint; scraped by Prometheus.

Key Metrics (Gauges/Counters):
- `pipeline_runs_total`
- `pipeline_image_resolution_width`, `pipeline_image_resolution_height`
- `pipeline_objects_count`, `pipeline_num_segments`, `pipeline_num_object_types`
- `pipeline_avg_segment_resolution`
- `pipeline_type_of_object_counted{object_type}`
- `pipeline_objects_by_type{object_type}`
- `pipeline_classifier_min_confidence`, `pipeline_classifier_confidence{segment_index}`
- `pipeline_sam_inference_ms`, `pipeline_classifier_inference_ms`, `pipeline_zeroshot_inference_ms`, `pipeline_overall_response_ms`

---

## Notes
- Object labels and candidate classes are configured in `pipeline/config.py`.
- Saved overlays (panoptic maps) reside in `pipeline/outputs/`.
- End-to-end flow is also exposed via the web UI at `/` (template upload).
