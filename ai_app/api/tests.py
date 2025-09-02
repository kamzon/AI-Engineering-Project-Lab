from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APITestCase
from records.models import Result

class CorrectionViewTest(APITestCase):
    def setUp(self):
        self.result = Result.objects.create(
            image='.source/image.png',
            object_type='cat',
            status='predicted',
            predicted_count=3
        )
        self.url = reverse('correct')  

    def test_correction_view_success(self):
        data = {
            'result_id': self.result.id,
            'corrected_count': 10
        }
        response = self.client.post(self.url, data)
        self.result.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.result.corrected_count, 10)
        self.assertEqual(self.result.status, 'corrected')

    def test_correction_view_missing_fields(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)

        def test_correction_view_non_integer(self):
            data = {
                'result_id': self.result.id,
                'corrected_count': 'not_a_number'
            }
            response = self.client.post(self.url, data)
            self.assertEqual(response.status_code, 400)


    class CountViewTest(APITestCase):
        def setUp(self):
            self.url = reverse('count')  # Update with your actual URL name

        def test_count_view_missing_fields(self):
            response = self.client.post(self.url, {})
            self.assertEqual(response.status_code, 400)

        def test_count_view_missing_image(self):
            data = {'object_type': 'cat'}
            response = self.client.post(self.url, data)
            self.assertEqual(response.status_code, 400)

        def test_count_view_missing_object_type(self):
            with open('source/image.png', 'rb') as img:
                data = {'image': img}
                response = self.client.post(self.url, data)
            self.assertEqual(response.status_code, 400)

        # To test success, you need a valid image and a working pipeline. This is usually done with mocks.
        # Example:
        # from unittest.mock import patch
        # @patch('pipeline.model.SamSegmentationClassifier.run')
        # def test_count_view_success(self, mock_run):
        #     mock_run.return_value = {
        #         'label_counts': {'cat': 5},
        #         'meta': {'info': 'test'}
        #     }
        #     with open('source/image.png', 'rb') as img:
        #         data = {'image': img, 'object_type': 'cat'}
        #         response = self.client.post(self.url, data)
        #     self.assertEqual(response.status_code, 201)
        #     self.assertIn('predicted_count', response.data)