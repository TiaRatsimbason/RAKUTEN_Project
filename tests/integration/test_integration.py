import unittest
import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.admin_token = self.get_token("admin", "admin_password")
        self.user_token = self.get_token("user", "user_password")
    
    def get_token(self, username, password):
        response = requests.post(f"{BASE_URL}/api/token", auth=HTTPBasicAuth(username, password))
        return response.json().get("access_token")
    
    def test_train_model(self):
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = requests.post(f"{BASE_URL}/api/train-model/", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model training started", response.json().get("message"))
    
    def test_predict(self):
        headers = {"Authorization": f"Bearer {self.user_token}"}
        files = {'file': open('test_image.jpg', 'rb')}
        data = {'n_samples': 10}
        response = requests.post(f"{BASE_URL}/api/predict/", headers=headers, files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("category", response.json())
    
    def test_unauthorized_access(self):
        # A user should not be able to access the train-model endpoint
        headers = {"Authorization": f"Bearer {self.user_token}"}
        response = requests.post(f"{BASE_URL}/api/train-model/", headers=headers)
        self.assertEqual(response.status_code, 403)

if __name__ == '__main__':
    unittest.main()
