import unittest
import requests
import time

BASE_URL = "http://localhost:8000"

class TestPerformance(unittest.TestCase):

    def test_prediction_response_time(self):
        files = {'file': open('data/preprocessed/image_test/image_529140_product_923202.jpg', 'rb')}
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/predict/", files=files)
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 0.2)  # 200 ms max
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
