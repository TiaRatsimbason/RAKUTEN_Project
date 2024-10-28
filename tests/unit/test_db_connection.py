# test_db_connection.py
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from db.db_connection import get_database


class TestDatabaseConnection(unittest.TestCase):

    @patch("db.db_connection.MongoClient")
    def test_get_database_success(self, mock_mongo_client):
        # Simule un client MongoDB fictif
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        # Simule la base de données par défaut
        mock_db = MagicMock()
        mock_client_instance.get_default_database.return_value = mock_db

        # Appelle la fonction get_database
        db = get_database()

        # Vérifie que MongoClient a été appelé avec l'URI par défaut
        mock_mongo_client.assert_called_with("mongodb://localhost:27017/")

        # Vérifie que get_default_database a été appelé
        mock_client_instance.get_default_database.assert_called_once()

        # Vérifie que la base de données retournée est celle simulée
        self.assertEqual(db, mock_db)

    @patch("db.db_connection.MongoClient")
    def test_get_database_with_custom_uri(self, mock_mongo_client):
        # Définir un URI personnalisé
        custom_uri = "mongodb://custom_host:27018/"

        # Remplace la variable d'environnement MONGODB_URI
        with patch("os.getenv", return_value=custom_uri):
            db = get_database()

            # Vérifie que MongoClient a été appelé avec l'URI personnalisé
            mock_mongo_client.assert_called_with(custom_uri)

    @patch("db.db_connection.MongoClient")
    def test_get_database_failure(self, mock_mongo_client):
        # Simule une erreur de connexion à MongoDB
        mock_mongo_client.side_effect = Exception("Connection failed")

        # Vérifie que l'exception est bien levée en cas d'échec
        with self.assertRaises(Exception) as context:
            get_database()
        self.assertEqual(str(context.exception), "Connection failed")


if __name__ == "__main__":
    unittest.main()
