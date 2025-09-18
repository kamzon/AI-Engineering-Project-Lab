from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient
from .models import Result


class RecordsViewSetTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        # Seed some results
        Result.objects.create(image="uploads/2025/09/18/a.png", object_type="cat", predicted_count=1)
        Result.objects.create(image="uploads/2025/09/18/b.png", object_type="car", predicted_count=2)

    def test_list_records(self):
        url = "/api/records/"
        # Disable static token middleware for this test
        with override_settings(API_STATIC_TOKEN=""):
            resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.data, list)
        self.assertGreaterEqual(len(resp.data), 2)
