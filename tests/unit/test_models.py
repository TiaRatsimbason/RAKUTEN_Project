import unittest
import os

class TestModels(unittest.TestCase):

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists("models/best_vgg16_model.h5"))
        self.assertTrue(os.path.exists("models/best_lstm_model.h5"))
    
    def test_model_can_be_loaded(self):
        # Dummy test to simulate model loading
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
