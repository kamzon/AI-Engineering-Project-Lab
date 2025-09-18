from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from records.models import Result
from .models import ThemePreference
from django.core.files.base import ContentFile
from pathlib import Path


class WebViewsTest(TestCase):
    def setUp(self):
        # Create a minimal Result so index/history have content
        self.result = Result.objects.create(
            image="uploads/2025/09/18/test.png",
            object_type="cat",
            predicted_count=2,
            status="predicted",
            meta={"note": "ok"},
        )

    def test_index_renders(self):
        url = reverse("home")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("object_types", resp.context)

    def test_history_json(self):
        url = reverse("history")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)
        # At least one record present
        self.assertGreaterEqual(len(resp.json()), 1)
        self.assertIn("corrections_allowed", resp.json()[0])

    def test_set_theme_valid(self):
        url = reverse("set_theme")
        resp = self.client.post(url, {"theme": "dracula"})
        self.assertEqual(resp.status_code, 200)
        pref = ThemePreference.objects.first()
        self.assertIsNotNone(pref)
        self.assertEqual(pref.theme, "dracula")

    def test_set_theme_invalid(self):
        url = reverse("set_theme")
        resp = self.client.post(url, {"theme": "invalid_theme"})
        self.assertEqual(resp.status_code, 400)
