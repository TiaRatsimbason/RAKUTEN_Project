import requests
from unittest import TestCase

BASE_URL = "http://localhost:8000"

class TestMetrics(TestCase):

    def test_model_metrics(self):
        response = requests.get(f"{BASE_URL}/api/evaluate-model")
        self.assertEqual(response.status_code, 200)
