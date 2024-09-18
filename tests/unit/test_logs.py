import unittest
import os

class TestLogs(unittest.TestCase):

    def test_train_log_created(self):
        self.assertTrue(os.path.exists("logs/train.log"))
    
    def test_validation_log_created(self):
        self.assertTrue(os.path.exists("logs/validation.log"))

if __name__ == '__main__':
    unittest.main()
