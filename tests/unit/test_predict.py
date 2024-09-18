import requests
from unittest import TestCase

BASE_URL = "http://localhost:8000"

class TestPredict(TestCase):

    def test_predict(self):
        files = {'file': open('data/preprocessed/image_test/image_529140_product_923202.jpg', 'rb')}
        response = requests.post(f"{BASE_URL}/api/predict/", files=files)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
