import unittest
import requests

BASE_URL = "http://localhost:8000"

class TestPredict(unittest.TestCase):

    def test_predict(self):
        files = {'file': open('test_image.jpg', 'rb')}
        response = requests.post(f"{BASE_URL}/api/predict/", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("category", response.json())

if __name__ == '__main__':
    unittest.main()
