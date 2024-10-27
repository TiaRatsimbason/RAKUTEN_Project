import sys
import os
import pytest
import pandas as pd
from unittest import mock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import numpy as np
from src.api.app import app
from PIL import Image
import io

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
    from src.scripts import predict

    # Mocking mlflow.tracking.MlflowClient
    mock_mlflow_client = mocker.patch("src.scripts.predict.mlflow.tracking.MlflowClient")
    mock_client_instance = mock_mlflow_client.return_value
    mock_client_instance.get_model_version.return_value = mocker.Mock(run_id='test_run_id')
    mock_client_instance.get_run.return_value = mocker.Mock(info=mocker.Mock(experiment_id='test_experiment_id'))

    # Mocking os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mocking open to return a valid image for image paths
    def mock_open_image(file, *args, **kwargs):
        if 'image' in file:
            # Créer une petite image factice en mémoire
            image = Image.new('RGB', (224, 224), color='white')
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            return mock.mock_open(read_data=img_byte_arr.read()).return_value
        elif 'tokenizer_config.json' in file:
            return mock.mock_open(read_data=b'{"config": "tokenizer"}').return_value
        elif 'best_weights.json' in file:
            return mock.mock_open(read_data=b'{"weights": [0.5, 0.5]}').return_value
        elif 'mapper.json' in file:
            return mock.mock_open(read_data=b'{"0": "10", "1": "20"}').return_value
        else:
            return mock.mock_open(read_data=b'').return_value

    mocker.patch("builtins.open", side_effect=mock_open_image)

    # Mocking keras.models.load_model with fake models
    mock_lstm_model = mocker.Mock()
    mock_lstm_model.predict.return_value = np.array([[0.8, 0.2], [0.4, 0.6]])  # Simulate two predictions
    mock_vgg16_model = mocker.Mock()
    mock_vgg16_model.predict.return_value = np.array([[0.3, 0.7], [0.5, 0.5]])  # Simulate two predictions
    mocker.patch("src.scripts.predict.keras.models.load_model", side_effect=[mock_lstm_model, mock_vgg16_model])

    # Mocking tokenizer_from_json and texts_to_sequences
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]  # Simulate tokenized output
    mocker.patch("src.scripts.predict.tokenizer_from_json", return_value=mock_tokenizer)

    # Create a real Predict instance to capture logs of best_weights
    predict_instance = predict.Predict(mock_tokenizer, mock_lstm_model, mock_vgg16_model, [0.5, 0.5], {"0": "10", "1": "20"})
    mocker.patch("src.scripts.predict.Predict", return_value=predict_instance)

    # Mocking DataImporter and load_data
    mock_importer = mocker.patch("src.api.routes.model.DataImporter")
    mock_importer_instance = mock_importer.return_value
    df_mock = pd.DataFrame({
        "description": ["item1desc", "item2desc"],
        "image_path": ["path/to/fake_image_1.jpg", "path/to/fake_image_2.jpg"],
        "prdtypecode": [0, 1]  # Populate with non-empty values for y_eval
    })
    mock_importer_instance.load_data.return_value = df_mock

    # Mock split_train_test to provide X_eval_sample and y_eval_sample for evaluation
    def mock_split_train_test(df, **kwargs):
        X_eval_mock = df[["description", "image_path"]]
        y_eval_mock = df["prdtypecode"]
        X_eval_sample = X_eval_mock.iloc[:2]  # Select two rows for evaluation
        y_eval_sample = y_eval_mock.iloc[:2]

        # Debugging output to verify non-empty samples
        print(f"Debug X_eval_sample in mock_split_train_test: {X_eval_sample}")
        print(f"Debug y_eval_sample in mock_split_train_test: {y_eval_sample}")

        return None, None, X_eval_sample, None, None, y_eval_sample

    mock_importer_instance.split_train_test.side_effect = mock_split_train_test

    # Mock TextPreprocessor and ImagePreprocessor
    mocker.patch("src.scripts.predict.TextPreprocessor")
    mocker.patch("src.scripts.predict.ImagePreprocessor")

    # Mocking load_data to ensure X_eval_sample and y_eval_sample are passed correctly
    _, _, X_eval_sample, _, _, y_eval_sample = mock_split_train_test(df_mock)

    # Assertion to verify the non-emptiness of X_eval_sample and y_eval_sample before proceeding
    assert not X_eval_sample.empty, "X_eval_sample is empty!"
    assert not y_eval_sample.empty, "y_eval_sample is empty!"

    # Manually override y_eval_sample if needed to ensure it is not empty
    if y_eval_sample.empty:
        y_eval_sample = pd.Series([0, 1])
        print("Debug: Manually set y_eval_sample to avoid empty value.")

    # Mock the evaluate_model function to use the non-empty X_eval_sample and y_eval_sample
    mock_evaluate = mocker.patch("src.api.routes.model.evaluate_model")
    mock_evaluate.return_value = {"evaluation_report": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85}}

    # Call the API to evaluate the model
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/model/evaluate-model/", params={"version": 1})

    # Check the response
    assert response.status_code == 200, f"Erreur: {response.status_code}, Détails: {response.text}"
    response_json = response.json()["evaluation_report"]
    assert "precision" in response_json
    assert "recall" in response_json
    assert "f1-score" in response_json