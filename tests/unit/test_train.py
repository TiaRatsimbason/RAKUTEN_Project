import unittest
import requests

BASE_URL = "http://localhost:8000"

class TestTrain(unittest.TestCase):

    def test_train_model(self):
        response = requests.post(f"{BASE_URL}/api/train-model/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model training started", response.json().get("message"))

if __name__ == '__main__':
    unittest.main()
