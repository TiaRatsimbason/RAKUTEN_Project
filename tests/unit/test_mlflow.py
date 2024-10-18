import unittest
from unittest.mock import patch, MagicMock
from train_model import TextLSTMModel, ImageVGG16Model
import pandas as pd
import numpy as np

class TestMLFlowLogging(unittest.TestCase):

    @patch("mlflow.start_run")
    @patch("mlflow.end_run")
    @patch("mlflow.log_param")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_artifact")
    @patch("mlflow.tensorflow.log_model")
    def test_text_lstm_mlflow_logging(self, mock_log_model, mock_log_artifact, mock_log_metric, mock_log_param, mock_end_run, mock_start_run):
        # Arrange
        X_train = pd.DataFrame({"description": ["This is a product", "Another product"]})
        y_train = pd.Series([1, 0])
        X_val = pd.DataFrame({"description": ["Validation product"]})
        y_val = pd.Series([1])
        
        model = TextLSTMModel()

        # Act
        model.preprocess_and_fit(X_train, y_train, X_val, y_val)

        # Assert that mlflow was used to log parameters, metrics, and artifacts
        mock_start_run.assert_called_once()
        mock_log_param.assert_any_call("max_words", 10000)
        mock_log_param.assert_any_call("max_sequence_length", 10)
        mock_log_metric.assert_any_call("accuracy", unittest.mock.ANY)
        mock_log_artifact.assert_called_with("models/tokenizer_config.json")
        mock_log_model.assert_called_once()
        mock_end_run.assert_called_once()

    @patch("mlflow.start_run")
    @patch("mlflow.end_run")
    @patch("mlflow.log_param")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_artifact")
    @patch("mlflow.tensorflow.log_model")
    def test_image_vgg16_mlflow_logging(self, mock_log_model, mock_log_artifact, mock_log_metric, mock_log_param, mock_end_run, mock_start_run):
        # Arrange
        X_train = pd.DataFrame({"image_path": ["path/to/image1.jpg", "path/to/image2.jpg"]})
        y_train = pd.Series([1, 0])
        X_val = pd.DataFrame({"image_path": ["path/to/image3.jpg"]})
        y_val = pd.Series([1])

        model = ImageVGG16Model()

        # Act
        model.preprocess_and_fit(X_train, y_train, X_val, y_val)

        # Assert that mlflow was used to log parameters, metrics, and artifacts
        mock_start_run.assert_called_once()
        mock_log_param.assert_any_call("batch_size", 23)
        mock_log_metric.assert_any_call("accuracy", unittest.mock.ANY)
        mock_log_artifact.assert_called_with("logs")
        mock_log_model.assert_called_once()
        mock_end_run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
