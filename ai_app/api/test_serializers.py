from django.test import SimpleTestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from pathlib import Path
from .serializers import CountRequestSerializer, CorrectionRequestSerializer


BASE_INPUTS_DIR = (Path(__file__).resolve().parent.parent / "static" / "inputs")


def load_bytes(filename: str) -> bytes:
    return (BASE_INPUTS_DIR / filename).read_bytes()


class CountRequestSerializerTests(SimpleTestCase):
    def test_requires_image_or_images(self):
        s = CountRequestSerializer(data={"object_type": "cat"})
        self.assertFalse(s.is_valid())
        self.assertIn("Provide either 'image' or 'images'", str(s.errors))

    def test_validate_image_invalid_file(self):
        bad = SimpleUploadedFile("bad.txt", b"not an image", content_type="text/plain")
        s = CountRequestSerializer(data={"object_type": "cat", "image": bad})
        self.assertFalse(s.is_valid())
        self.assertIn("Upload a valid image", str(s.errors))

    def test_validate_images_list_with_index_error(self):
        good = SimpleUploadedFile("cat.png", load_bytes("cat.png"), content_type="image/png")
        bad = SimpleUploadedFile("bad.txt", b"xxx", content_type="text/plain")
        s = CountRequestSerializer(data={"object_type": "cat", "images": [good, bad]})
        self.assertFalse(s.is_valid())
        # Should indicate which index failed
        self.assertIn("index", str(s.errors))

    @override_settings(IMAGE_UPLOAD_MAX_WIDTH=128, IMAGE_UPLOAD_MAX_HEIGHT=128)
    def test_resolution_out_of_range(self):
        # Use a large image that will violate max size
        big = SimpleUploadedFile("big_resolution.jpg", load_bytes("big_resolution.jpg"), content_type="image/jpeg")
        s = CountRequestSerializer(data={"object_type": "cat", "image": big})
        self.assertFalse(s.is_valid())
        self.assertIn("Image resolution out of allowed range", str(s.errors))


class CorrectionRequestSerializerTests(SimpleTestCase):
    def test_negative_correction_rejected(self):
        s = CorrectionRequestSerializer(data={"result_id": 1, "corrected_count": -1})
        self.assertFalse(s.is_valid())
        self.assertIn("Ensure this value is greater than or equal to 0", str(s.errors))


