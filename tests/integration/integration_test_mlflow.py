import unittest
from fastapi.testclient import TestClient
from main import app

class TestMLFlowIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_train_model_endpoint(self):
        response = self.client.post("/api/model/train-model/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model training completed successfully", response.json()["message"])

    def test_evaluate_model_endpoint(self):
        response = self.client.post("/api/model/evaluate-model/")
        self.assertEqual(response.status_code, 200)
        evaluation_report = response.json()["evaluation_report"]
        self.assertTrue("precision" in evaluation_report)
        self.assertTrue("recall" in evaluation_report)
        self.assertTrue("f1-score" in evaluation_report)

if __name__ == "__main__":
    unittest.main()
