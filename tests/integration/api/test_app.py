import sys
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import numpy as np

# Ajouter dynamiquement src au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
sys.path.append(project_root)

# Moquer les fonctions MLflow avant l'importation de l'application
with patch('mlflow.set_tracking_uri'), \
     patch('mlflow.set_experiment'), \
     patch('mlflow.start_run'), \
     patch('mlflow.log_param'), \
     patch('mlflow.log_artifact'), \
     patch('mlflow.log_metric'), \
     patch('mlflow.register_model'):
    # Importer l'application FastAPI
    from src.api.app import app

# Importer les modules nécessaires
from src.scripts import predict, main

# Utiliser TestClient pour tester les routes FastAPI de manière synchrone
client = TestClient(app)

@pytest.mark.asyncio
async def test_train_model(mocker):
    # Mocking the train_and_save_model function to avoid real execution during tests
    mocker.patch('src.scripts.main.train_and_save_model', return_value=None)

    # Call the API to train the model
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/model/train-model/")

    # Check that the response is correct
    assert response.status_code == 200, f"Erreur: {response.status_code}, Détails: {response.text}"
    assert response.json() == {"message": "Model training completed successfully."}

@pytest.mark.asyncio
async def test_predict(mocker):
    # Mocking mlflow.tracking.MlflowClient
    mock_mlflow_client = mocker.patch("src.scripts.predict.mlflow.tracking.MlflowClient")
    mock_client_instance = mock_mlflow_client.return_value
    # Mocking get_model_version
    mock_model_version_details = mocker.Mock(run_id='test_run_id')
    mock_client_instance.get_model_version.return_value = mock_model_version_details
    # Mocking get_run
    mock_run_info = mocker.Mock(info=mocker.Mock(experiment_id='test_experiment_id'))
    mock_client_instance.get_run.return_value = mock_run_info

    # Mocking os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mocking open to return dummy data
    def mock_open_read_data(file, *args, **kwargs):
        if 'tokenizer_config.json' in file:
            return mocker.mock_open(read_data='{"config": "tokenizer"}').return_value
        elif 'best_weights.json' in file:
            return mocker.mock_open(read_data='{"weights": [0.5, 0.5]}').return_value
        elif 'mapper.json' in file:
            return mocker.mock_open(read_data='{"0": 0, "1": 1}').return_value
        else:
            return mocker.mock_open(read_data='').return_value

    mocker.patch("builtins.open", side_effect=mock_open_read_data)

    # Mocking keras.models.load_model to return a mock model
    mock_lstm_model = mocker.Mock()
    mock_vgg16_model = mocker.Mock()
    mocker.patch("src.scripts.predict.keras.models.load_model", side_effect=[mock_lstm_model, mock_vgg16_model])

    # Mocking tokenizer_from_json
    mock_tokenizer = mocker.Mock()
    mocker.patch("src.scripts.predict.tokenizer_from_json", return_value=mock_tokenizer)

    # Mocking the Predict class
    mock_predict_instance = mocker.Mock()
    mocker.patch("src.scripts.predict.Predict", return_value=mock_predict_instance)
    # Mocking the predict method of Predict instance
    mock_predict_instance.predict.return_value = {"0": "prediction_1", "1": "prediction_2"}

    # Mocking pandas.read_csv to simulate reading the CSV
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame({
        "description": ["desc1", "desc2"],
        "image_path": ["path1", "path2"]
    }))

    # Call the API for prediction
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/model/predict/",
            params={"version": 1, "images_folder": "some_folder"}
        )

    # Check that the response is correct
    assert response.status_code == 200, f"Erreur: {response.status_code}, Détails: {response.text}"
    assert response.json() == {"predictions": {"0": "prediction_1", "1": "prediction_2"}}

@pytest.mark.asyncio
async def test_evaluate_model(mocker):
    # Import the predict module
    from src.scripts import predict

    # Mocking mlflow.tracking.MlflowClient
    mock_mlflow_client = mocker.patch("src.scripts.predict.mlflow.tracking.MlflowClient")
    mock_client_instance = mock_mlflow_client.return_value

    # Mocking get_model_version
    mock_model_version_details = mocker.Mock(run_id='test_run_id')
    mock_client_instance.get_model_version.return_value = mock_model_version_details

    # Mocking get_run
    mock_run_info = mocker.Mock(info=mocker.Mock(experiment_id='test_experiment_id'))
    mock_client_instance.get_run.return_value = mock_run_info

    # Mocking os.path.exists to toujours retourner True
    mocker.patch("os.path.exists", return_value=True)

    # Mocking open pour retourner des données fictives
    def mock_open_read_data(file, *args, **kwargs):
        if 'tokenizer_config.json' in file:
            return mocker.mock_open(read_data='{"config": "tokenizer"}').return_value
        elif 'best_weights.json' in file:
            return mocker.mock_open(read_data='{"weights": [0.5, 0.5]}').return_value
        elif 'mapper.json' in file:
            # Pour correspondre au mapping corrigé, mappez '0' à '0' etc., si nécessaire
            return mocker.mock_open(read_data='{"0": 0, "1": 1}').return_value
        else:
            return mocker.mock_open(read_data='').return_value

    mocker.patch("builtins.open", side_effect=mock_open_read_data)

    # Mocking keras.models.load_model pour retourner un modèle fictif
    mock_lstm_model = mocker.Mock()
    mock_vgg16_model = mocker.Mock()
    mocker.patch("src.scripts.predict.keras.models.load_model", side_effect=[mock_lstm_model, mock_vgg16_model])

    # Mocking la fonction tokenizer_from_json
    mock_tokenizer = mocker.Mock()
    mocker.patch("src.scripts.predict.tokenizer_from_json", return_value=mock_tokenizer)

    # Mocking la classe Predict
    mock_predict_instance = mocker.Mock()
    mocker.patch("src.scripts.predict.Predict", return_value=mock_predict_instance)

    # Mocking la méthode predict de l'instance Predict
    # Retourner une pandas Series avec le même nombre d'éléments que X_eval_sample
    def mock_predict(X, path):
        return pd.Series([0] * len(X))

    mock_predict_instance.predict.side_effect = mock_predict

    # Mocking DataImporter où il est réellement importé
    mock_importer = mocker.patch("src.api.routes.model.DataImporter")
    mock_importer_instance = mock_importer.return_value

    # Mocking load_data pour retourner un DataFrame avec les colonnes requises
    df_mock = pd.DataFrame({
        "description": [f"item{i}desc" for i in range(10)],
        "image_path": [f"path{i}" for i in range(10)],
        "prdtypecode": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    # S'assurer que 'prdtypecode' est de type int
    df_mock['prdtypecode'] = df_mock['prdtypecode'].astype(int)

    mock_importer_instance.load_data.return_value = df_mock

    # Mocking split_train_test pour retourner des données fictives
    X_eval_mock = df_mock.drop(['prdtypecode'], axis=1)
    y_eval_mock = df_mock['prdtypecode']

    def mock_split_train_test(df, samples_per_class=1, val_samples_per_class=1):
        return None, None, X_eval_mock, None, None, y_eval_mock

    mock_importer_instance.split_train_test.side_effect = mock_split_train_test

    # Mocking TextPreprocessor et ImagePreprocessor où ils sont importés
    mock_text_preprocessor = mocker.patch("src.scripts.predict.TextPreprocessor")
    mock_image_preprocessor = mocker.patch("src.scripts.predict.ImagePreprocessor")

    # Appeler l'API pour évaluer le modèle
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/model/evaluate-model/", params={"version": 1})

    # Imprimer la réponse pour le débogage
    print(f"Réponse de l'API : {response.json()}")

    # Vérifier que la réponse est correcte
    assert response.status_code == 200, f"Erreur: {response.status_code}, Détails: {response.text}"
    response_json = response.json()
    assert "evaluation_report" in response_json
    assert "precision" in response_json["evaluation_report"]
    assert "recall" in response_json["evaluation_report"]
    assert "f1-score" in response_json["evaluation_report"]