import unittest
import requests

BASE_URL = "http://localhost:8000"

class TestMetrics(unittest.TestCase):

    def test_model_metrics(self):
        response = requests.get(f"{BASE_URL}/api/evaluate-model")
        self.assertEqual(response.status_code, 200)
        self.assertIn("accuracy", response.json())

if __name__ == '__main__':
    unittest.main()
