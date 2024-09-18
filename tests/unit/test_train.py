import requests
from unittest import TestCase

BASE_URL = "http://localhost:8000"

class TestTrain(TestCase):

    def test_train_model(self):
        response = requests.post(f"{BASE_URL}/api/train-model/")
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
