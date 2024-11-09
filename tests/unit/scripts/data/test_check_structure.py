# test_check_structure.py
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

from src.scripts.data.check_structure import check_existing_file, check_existing_folder


class TestCheckStructure(unittest.TestCase):

    @patch("builtins.input", return_value="y")
    @patch("os.path.isfile")
    def test_check_existing_file_exists(self, mock_isfile, mock_input):
        # Simule l'existence d'un fichier et la réponse "y" à l'invite
        mock_isfile.return_value = True
        result = check_existing_file("existing_file.txt")
        self.assertTrue(result)

    @patch("os.path.isfile")
    def test_check_existing_file_not_exists(self, mock_isfile):
        # Simule l'absence d'un fichier
        mock_isfile.return_value = False
        result = check_existing_file("non_existing_file.txt")
        self.assertTrue(result)

    @patch("builtins.input", return_value="y")
    @patch("os.path.exists")
    def test_check_existing_folder_exists(self, mock_exists, mock_input):
        # Simule l'existence d'un dossier et la réponse "y" à l'invite
        mock_exists.return_value = True
        result = check_existing_folder("existing_folder")
        self.assertFalse(result)

    @patch("builtins.input", return_value="y")
    @patch("os.path.exists")
    def test_check_existing_folder_not_exists(self, mock_exists, mock_input):
        # Simule l'absence d'un dossier et la réponse "y" à l'invite
        mock_exists.return_value = False
        result = check_existing_folder("non_existing_folder")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
