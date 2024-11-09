<<<<<<< HEAD
# test_import_raw_data.py
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))
from src.scripts.data.import_raw_data import import_raw_data


class TestImportRawData(unittest.TestCase):

    @patch("os.makedirs", side_effect=lambda x: None)  # Remplace les appels réels sans vérifier les chemins
    @patch("requests.get")
    @patch("src.scripts.data.import_raw_data.check_existing_folder", return_value=True)
    @patch("src.scripts.data.import_raw_data.check_existing_file", return_value=True)
    def test_import_raw_data_success(self, mock_check_folder, mock_check_file, mock_requests_get, mock_makedirs):
        # Simule une réponse de requête réussie
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"file content"
        mock_requests_get.return_value = mock_response

        # Appelle la fonction avec des valeurs de chemin simulées
        import_raw_data("data/raw", ["file1.csv", "file2.csv"], "https://bucket.url/")

        # Vérifie que les requêtes ont été appelées au moins une fois
        assert mock_requests_get.call_count >= 1

        # Vérifie que makedirs a été appelé au moins deux fois
        assert mock_makedirs.call_count >= 2


if __name__ == "__main__":
    unittest.main()
=======
# test_import_raw_data.py
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))
from src.scripts.data.import_raw_data import import_raw_data


class TestImportRawData(unittest.TestCase):

    @patch("os.makedirs", side_effect=lambda x: None)  # Remplace les appels réels sans vérifier les chemins
    @patch("requests.get")
    @patch("src.scripts.data.import_raw_data.check_existing_folder", return_value=True)
    @patch("src.scripts.data.import_raw_data.check_existing_file", return_value=True)
    def test_import_raw_data_success(self, mock_check_folder, mock_check_file, mock_requests_get, mock_makedirs):
        # Simule une réponse de requête réussie
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"file content"
        mock_requests_get.return_value = mock_response

        # Appelle la fonction avec des valeurs de chemin simulées
        import_raw_data("data/raw", ["file1.csv", "file2.csv"], "https://bucket.url/")

        # Vérifie que les requêtes ont été appelées au moins une fois
        assert mock_requests_get.call_count >= 1

        # Vérifie que makedirs a été appelé au moins deux fois
        assert mock_makedirs.call_count >= 2

    @patch("requests.get")
    def test_import_raw_data_failure(self, mock_requests_get):
        # Simule une requête échouée (erreur 404)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        # Vérifie qu'une exception est levée si le fichier est introuvable
        with self.assertRaises(Exception):
            import_raw_data("data/raw", ["missing_file.csv"], "https://bucket.url/")


if __name__ == "__main__":
    unittest.main()
>>>>>>> cdb26b740 (essai d'enregistrement des données)
