import unittest
import requests

BASE_URL = "http://localhost:8000"

class TestAPI(unittest.TestCase):
    
    def test_health_check(self):
        response = requests.get(f"{BASE_URL}/docs")
        self.assertEqual(response.status_code, 200)
    
    def test_setup_data(self):
        response = requests.post(f"{BASE_URL}/api/setup-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Data setup complete", response.json().get("message"))
    
    def test_train_model(self):
        response = requests.post(f"{BASE_URL}/api/train-model/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model training started", response.json().get("message"))

    def test_predict(self):
        files = {'file': open('test_image.jpg', 'rb')}
        data = {'n_samples': 5}
        response = requests.post(f"{BASE_URL}/api/predict/", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("category", response.json())

if __name__ == '__main__':
    unittest.main()
