# test_make_dataset.py
import os
import sys
import unittest
from unittest.mock import patch

# Ajuste le chemin pour l'importation du script à tester
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))
from src.scripts.data.make_dataset import main


class TestMakeDataset(unittest.TestCase):

    @patch("shutil.rmtree")
    @patch("shutil.copytree")
    @patch("os.path.exists", return_value=True)  # Simule l'existence du dossier de destination
    @patch("click.Path.convert", side_effect=lambda value, param, ctx: value)  # Désactive la vérification de Click
    @patch("sys.argv",
           ["make_dataset.py", "source_folder", "destination_folder"])  # Simule les arguments de ligne de commande
    def test_make_dataset_with_existing_output(self, mock_path_convert, mock_exists, mock_copytree, mock_rmtree):
        # Capture SystemExit pour éviter l'arrêt du test
        with self.assertRaises(SystemExit) as cm:
            main()

        # Vérifie que rmtree est appelé pour supprimer le dossier existant
        mock_rmtree.assert_called_once_with("destination_folder")
        # Vérifie que copytree est appelé pour copier le contenu du dossier source
        mock_copytree.assert_called_once_with("source_folder", "destination_folder")
        # Vérifie que l'exception est levée avec un code de sortie 0
        self.assertEqual(cm.exception.code, 0)

    @patch("shutil.copytree")
    @patch("os.path.exists", return_value=False)  # Simule l'absence du dossier de destination
    @patch("click.Path.convert", side_effect=lambda value, param, ctx: value)  # Désactive la vérification de Click
    @patch("sys.argv",
           ["make_dataset.py", "source_folder", "destination_folder"])  # Simule les arguments de ligne de commande
    def test_make_dataset_without_existing_output(self, mock_path_convert, mock_exists, mock_copytree):
        # Capture SystemExit pour éviter l'arrêt du test
        with self.assertRaises(SystemExit) as cm:
            main()

        # Vérifie que copytree est appelé pour copier le contenu du dossier source
        mock_copytree.assert_called_once_with("source_folder", "destination_folder")
        # Vérifie que l'exception est levée avec un code de sortie 0
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
