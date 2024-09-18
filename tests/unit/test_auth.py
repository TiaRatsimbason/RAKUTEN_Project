import unittest
import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"

class TestAuth(unittest.TestCase):

    def test_register_user(self):
        payload = {"username": "new_user", "password": "password123", "role": "user"}
        response = requests.post(f"{BASE_URL}/api/register", json=payload)
        self.assertEqual(response.status_code, 201)
    
    def test_get_token(self):
        response = requests.post(f"{BASE_URL}/api/token", auth=HTTPBasicAuth("admin", "admin_password"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
    
    def test_unauthorized_access(self):
        # Try accessing an admin route without being authorized
        headers = {"Authorization": "Bearer invalid_token"}
        response = requests.post(f"{BASE_URL}/api/train-model/", headers=headers)
        self.assertEqual(response.status_code, 403)

if __name__ == '__main__':
    unittest.main()
