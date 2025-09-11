from django.urls import reverse
from django.test import override_settings
from rest_framework.test import APITestCase
from unittest.mock import patch
from django.core.files.uploadedfile import SimpleUploadedFile
from records.models import Result
from PIL import Image
import io
import tempfile
import os
from pathlib import Path


def _make_test_image_file(name: str = "test.png", size=(64, 64)) -> SimpleUploadedFile:
    """Create an in-memory PNG image file for upload tests."""
    img = Image.new("RGB", size, color=(123, 222, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return SimpleUploadedFile(name, buf.getvalue(), content_type="image/png")


class CorrectionViewTest(APITestCase):
    def setUp(self):
        # Minimal Result entry; image path field can be empty for this test
        self.result = Result.objects.create(
            image="uploads/test.png",
            object_type="cat",
            status="predicted",
            predicted_count=3,
        )
        self.url = reverse("correct")

    def test_correction_view_success(self):
        data = {"result_id": self.result.id, "corrected_count": 10}
        response = self.client.post(self.url, data)
        self.result.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.result.corrected_count, 10)
        self.assertEqual(self.result.status, "corrected")

    def test_correction_view_missing_fields(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)

    def test_correction_view_non_integer(self):
        data = {"result_id": self.result.id, "corrected_count": "not_a_number"}
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, 400)


class CountViewTest(APITestCase):
    def setUp(self):
        self.url = reverse("count")

    def test_count_view_missing_fields(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)

    def test_count_view_missing_image(self):
        data = {"object_type": "cat"}
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, 400)

    def test_count_view_missing_object_type(self):
        img_file = _make_test_image_file()
        data = {"image": img_file}
        response = self.client.post(self.url, data, format="multipart")
        self.assertEqual(response.status_code, 400)

    @patch("pipeline.pipeline.Pipeline.run")
    def test_count_view_success(self, mock_run):
        # Prepare a temporary panoptic file so copying in the view succeeds
        with tempfile.TemporaryDirectory() as tmpdir:
            panoptic_tmp = os.path.join(tmpdir, "panoptic.png")
            with open(panoptic_tmp, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

            mock_run.return_value = {
                "predicted_classes": ["cat"],
                "zero_shot_labels": ["cat"],
                "label_counts": {"cat": 3, "other": 0},
                "id": "testid123",
                "panoptic_path": panoptic_tmp,
                "metadata": {
                    "image_resolution": {"width": 64, "height": 64},
                    "inference_time_ms_per_model": {"sam_ms": 1.2, "classifier_ms": 0.8, "overall_ms": 3.1},
                },
            }

            img_file = _make_test_image_file()
            data = {"object_type": "cat", "image": img_file}

            # Isolate media writes to a temp directory
            # MEDIA_ROOT as Path to be compatible with Path-style joins inside the view
            with override_settings(MEDIA_ROOT=Path(tmpdir), MEDIA_URL="/media/"):
                response = self.client.post(self.url, data, format="multipart")

        self.assertEqual(response.status_code, 201)
        self.assertIn("predicted_count", response.data)
        self.assertEqual(response.data.get("predicted_count"), 3)


class GenerateViewTest(APITestCase):
    def setUp(self):
        self.url = reverse("generate")

    def test_generate_missing_fields(self):
        # Empty payload should fail serializer validation
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 400)

    @patch("api.views.augment_image")
    @patch("api.views.generate_image_with_api")
    @patch("pipeline.models.few_shot.FewShotResNet")
    def test_generate_success(self, mock_fewshot_cls, mock_gen, mock_aug):
        # Mock image generator to return a small PIL image
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        mock_gen.return_value = img
        mock_aug.side_effect = lambda im, **kwargs: im

        # Mock FewShotResNet trainer
        trainer = mock_fewshot_cls.return_value
        trainer.load_data_from_paths.return_value = None
        trainer.finetune.return_value = None

        payload = {
            "num_images": 1,
            "max_objects_per_image": 1,
            "object_types": ["cat"],
            "backgrounds": ["random"],
            "blur": 0,
            "rotate": [0],
            "noise": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with override_settings(MEDIA_ROOT=Path(tmpdir), MEDIA_URL="/media/"):
                response = self.client.post(self.url, payload, format="json")

        self.assertEqual(response.status_code, 201)
        self.assertIn("positives", response.data)
        self.assertIn("negatives", response.data)
        self.assertIn("finetuned_model_dir", response.data)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(response.data.get("positives", 0), 0)