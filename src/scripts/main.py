import datetime
import json
import os
import keras
import mlflow
import numpy as np
import logging
from sklearn.metrics import accuracy_score
from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.scripts.models.train_model import TextLSTMModel, ImageVGG16Model, concatenate

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    try:
        logger.info("Starting model training")
        
        # Créer les dossiers nécessaires
        os.makedirs("/app/mlruns", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Configuration MLflow
        mlflow.set_tracking_uri("http://mlflow-ui:5000")
        os.environ['MLFLOW_TRACKING_URI'] = "http://mlflow-ui:5000"
        os.environ['MLFLOW_ENABLE_CORS'] = 'true'
        logger.info("MLflow tracking URI set")

        # Créer une expérience MLflow
        experiment_name = "Rakuten Model Training"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")

        # Timestamp unique pour le run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Démarrer le run MLflow
        with mlflow.start_run(run_name=f"Train Models {timestamp}") as main_run:
            # Chargement et préparation des données
            logger.info("Loading and preparing data")
            data_importer = DataImporter()
            df = data_importer.load_data()
            X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

            # Prétraitement
            logger.info("Preprocessing text and images")
            text_preprocessor = TextPreprocessor()
            image_preprocessor = ImagePreprocessor()
            
            text_preprocessor.preprocess_text_in_df(X_train, ["description"])
            text_preprocessor.preprocess_text_in_df(X_val, ["description"])
            
            image_preprocessor.preprocess_images_in_df(X_train)
            image_preprocessor.preprocess_images_in_df(X_val)

            # Log des paramètres globaux
            mlflow.log_param("num_classes", 27)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("max_sequence_length", 10)

            # Entraînement LSTM
            logger.info("Training LSTM model")
            text_lstm_model = TextLSTMModel()
            text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

            # Sauvegarde LSTM
            lstm_checkpoint_path = os.path.join("models", "best_lstm_model.keras")
            text_lstm_model.model.save(lstm_checkpoint_path)
            mlflow.log_artifact(lstm_checkpoint_path)

            # Sauvegarde du tokenizer
            tokenizer_path = os.path.join("models", "tokenizer_config.json")
            with open(tokenizer_path, "w", encoding="utf-8") as json_file:
                json_file.write(text_lstm_model.tokenizer.to_json())
            mlflow.log_artifact(tokenizer_path)
            logger.info("LSTM model and tokenizer saved")

            # Entraînement VGG16
            logger.info("Training VGG16 model")
            image_vgg16_model = ImageVGG16Model()
            image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

            # Sauvegarde VGG16
            vgg16_checkpoint_path = os.path.join("models", "best_vgg16_model.keras")
            image_vgg16_model.model.save(vgg16_checkpoint_path)
            mlflow.log_artifact(vgg16_checkpoint_path)
            logger.info("VGG16 model saved")

            # Entraînement du modèle concaténé
            logger.info("Training concatenate model")
            model_concatenate = concatenate(
                text_lstm_model.tokenizer,
                text_lstm_model.model,
                image_vgg16_model.model
            )

            lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(
                X_train, y_train
            )
            best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
            logger.info(f"Best weights found: LSTM={best_weights[0]}, VGG16={best_weights[1]}")

            # Sauvegarde des poids optimaux
            best_weights_path = os.path.join("models", "best_weights.json")
            with open(best_weights_path, "w") as file:
                json.dump({
                    "lstm_weight": float(best_weights[0]),
                    "vgg16_weight": float(best_weights[1])
                }, file, indent=4)
            mlflow.log_artifact(best_weights_path)

            logger.info("Training completed successfully")
            return True

    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()