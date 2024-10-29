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

    # Mocking NLTK download calls
    mocker.patch("nltk.download", return_value=None)

    # Mocking MongoDB collection with insert_one
    mock_collection = mocker.Mock()
    mock_collection.insert_one.return_value = mocker.Mock()

    # Create a mock class for MongoDB client
    class MockMongoClient:
        def __init__(self, *args, **kwargs):
            pass
        
        def __getitem__(self, name):
            return {"model_evaluation": mock_collection}[name]

    # Patch MongoClient
    mocker.patch("src.api.routes.model.MongoClient", MockMongoClient)
    # Ensure the collection is available directly
    mocker.patch("src.api.routes.model.collection", mock_collection)

    # Mocking mlflow.tracking.MlflowClient
    mock_mlflow_client = mocker.patch("src.scripts.predict.mlflow.tracking.MlflowClient")
    mock_client_instance = mock_mlflow_client.return_value
    mock_client_instance.get_model_version.return_value = mocker.Mock(run_id='test_run_id')
    mock_client_instance.get_run.return_value = mocker.Mock(info=mocker.Mock(experiment_id='test_experiment_id'))

    # Mocking os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mocking open to return valid data
    def mock_open_image(file, *args, **kwargs):
        if 'image' in file:
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

    # Mocking keras models
    mock_lstm_model = mocker.Mock()
    mock_lstm_model.predict.return_value = np.array([[0.8, 0.2], [0.4, 0.6]])
    mock_vgg16_model = mocker.Mock()
    mock_vgg16_model.predict.return_value = np.array([[0.3, 0.7], [0.5, 0.5]])
    mocker.patch("src.scripts.predict.keras.models.load_model", side_effect=[mock_lstm_model, mock_vgg16_model])

    # Mocking tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    mocker.patch("src.scripts.predict.tokenizer_from_json", return_value=mock_tokenizer)

    # Create a mock DataFrame with test data - Using 10 samples
    df_mock = pd.DataFrame({
        "description": ["item1desc", "item2desc"] * 5,
        "image_path": [f"path/to/fake_image_{i}.jpg" for i in range(10)],
        "imageid": [f"img{i}" for i in range(10)],
        "productid": [f"prod{i}" for i in range(10)],
        "prdtypecode": [0, 1] * 5
    })

    # Mock DataImporter
    mock_importer = mocker.patch("src.api.routes.model.DataImporter")
    mock_importer_instance = mock_importer.return_value
    mock_importer_instance.load_data.return_value = df_mock

    # Mock split_train_test to return the full DataFrame for evaluation
    def mock_split_train_test(df, **kwargs):
        return None, None, df[["description", "image_path", "imageid", "productid"]], None, None, df["prdtypecode"]

    mock_importer_instance.split_train_test.side_effect = mock_split_train_test

    # Calculate expected sample size (30% of 10 = 3)
    expected_sample_size = int(len(df_mock) * 0.3)

    # Create a fixed sample for consistent testing
    sample_indices = [0, 1, 2]  # First 3 indices for deterministic testing
    expected_predictions = {str(i): i % 2 for i in range(expected_sample_size)}

    # Mock the sample method to always return the same indices
    def mock_sample(*args, **kwargs):
        result_df = df_mock.iloc[sample_indices][["description", "image_path", "imageid", "productid"]]
        # Ensure index is reset to avoid any potential issues
        return result_df.reset_index(drop=True)

    mocker.patch.object(pd.DataFrame, 'sample', side_effect=mock_sample)

    # Mock predictor with fixed predictions matching the sample size
    mock_predictor = mocker.Mock()
    mock_predictor.predict.return_value = expected_predictions
    mocker.patch("src.api.routes.model.load_predictor", return_value=mock_predictor)

    # Mock TextPreprocessor and ImagePreprocessor
    mocker.patch("src.scripts.predict.TextPreprocessor")
    mocker.patch("src.scripts.predict.ImagePreprocessor")

    # Call the API
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/model/evaluate-model/", params={"version": 1})

    # Check the response
    assert response.status_code == 200, f"Erreur: {response.status_code}, Détails: {response.text}"
    evaluation_report = response.json()["evaluation_report"]
    
    # Verify MongoDB interaction
    mock_collection.insert_one.assert_called_once()
    
    # Verify the structure of the evaluation report
    assert "precision" in evaluation_report
    assert "recall" in evaluation_report
    assert "f1-score" in evaluation_report
    
    # Verify the metrics are float values between 0 and 1
    assert 0 <= float(evaluation_report["precision"]) <= 1
    assert 0 <= float(evaluation_report["recall"]) <= 1
    assert 0 <= float(evaluation_report["f1-score"]) <= 1