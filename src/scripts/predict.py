import json
import logging
import os
import argparse

import keras
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.scripts.features.build_features import TextPreprocessor
from src.scripts.data.gridfs_image_handler import GridFSImageHandler
from src.config.mongodb import sync_db

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predict:
    def __init__(self, tokenizer, lstm, vgg16, best_weights, mapper):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.image_handler = None

    def predict(self, df):
        """
        Fait des prédictions en utilisant les images depuis GridFS
        """
        try:
            # Vérifier les données
            if df is None or len(df) == 0:
                raise ValueError("Empty or null DataFrame provided")
                
            required_text_columns = ["description"]
            missing_text_columns = [col for col in required_text_columns if col not in df.columns]
            if missing_text_columns:
                raise ValueError(f"Missing required text columns: {missing_text_columns}")

            # Vérifier les valeurs nulles
            null_counts = df[required_text_columns].isnull().sum()
            if null_counts.any():
                logger.warning(f"Found null values in text columns: \n{null_counts[null_counts > 0]}")
                
            # Créer une copie pour éviter de modifier les données originales
            df = df.copy()
            
            # Remplir les valeurs nulles
            df["description"] = df["description"].fillna("")

            # Prétraitement du texte
            logger.info("Preprocessing text data...")
            text_preprocessor = TextPreprocessor()
            text_preprocessor.preprocess_text_in_df(df, columns=["description"])

            # Préparer les séquences de texte
            logger.info("Preparing text sequences...")
            sequences = self.tokenizer.texts_to_sequences(df["description"])
            padded_sequences = pad_sequences(
                sequences, maxlen=10, padding="post", truncating="post"
            )

            # Traitement des images depuis GridFS
            logger.info("Loading and preprocessing images from GridFS...")
            images = []
            
            # Initialiser GridFSImageHandler
            self.image_handler = GridFSImageHandler()
            self.image_handler.__enter__()

            # Déterminer la méthode d'extraction des images
            if 'gridfs_file_id' in df.columns:
                # Utiliser gridfs_file_id pour extraire les images
                logger.info("Using gridfs_file_id to extract images")
                image_paths = self.image_handler.batch_extract_images(df)
                key_column = 'gridfs_file_id'
            elif 'productid' in df.columns and 'imageid' in df.columns:
                # Utiliser productid et imageid pour extraire les images
                logger.info("Using productid and imageid to extract images")
                image_paths = self.image_handler.batch_extract_images_by_ids(df)
                key_column = 'key'  # Clé formée de 'productid_imageid'
            else:
                raise ValueError("Required columns for image extraction are missing")

            logger.info(f"Retrieved {len(image_paths)} image paths")
            
            if not image_paths:
                raise ValueError("No image paths were retrieved from GridFS")
            
            # Prétraiter les images
            for idx, row in df.iterrows():
                if key_column == 'gridfs_file_id':
                    key = str(row['gridfs_file_id'])
                else:
                    key = f"{str(row['productid'])}_{str(row['imageid'])}"
                
                if key not in image_paths:
                    logger.warning(f"Image path not found for key {key}")
                    continue  
                
                image_path = image_paths[key]
                if not os.path.exists(image_path):
                    logger.warning(f"Image file not found at {image_path} for key {key}")
                    continue  
                
                try:
                    img = load_img(image_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    images.append(img_array)
                    logger.debug(f"Successfully processed image for key {key}")
                except Exception as e:
                    logger.error(f"Error processing image for key {key}: {e}")
                    continue  

            if not images:
                raise ValueError("No images were successfully processed")
            
            images = np.array(images)
            logger.info(f"Processed {len(images)} images successfully")
            
            # Faire les prédictions
            logger.info("Making predictions with LSTM model...")
            lstm_proba = self.lstm.predict(padded_sequences)

            logger.info("Making predictions with VGG16 model...")
            vgg16_proba = self.vgg16.predict(images)

            # Combiner les probabilités selon les poids         
            logger.info("Combining model predictions...")
            concatenate_proba = (
                self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
            )
            final_predictions = np.argmax(concatenate_proba, axis=1)
            
            # Mapper les résultats aux catégories
            logger.info("Mapping predictions to categories...")
            predictions = {}

            logger.debug(f"Mapper content: {self.mapper}")
            logger.debug(f"Raw predictions: {final_predictions}")

            # Les prédictions sont les classes directement
            for i, pred in enumerate(final_predictions):
                # Mapper l'indice de classe prédite au code de produit original
                category_code = self.mapper.get(str(pred), "Unknown")
                predictions[str(i)] = category_code
                logger.debug(f"Prediction {i}: class {pred}, category code {category_code}")

            logger.info(f"Final predictions: {predictions}")
            return predictions

        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

        finally:
            if self.image_handler:
                try:
                    self.image_handler.__exit__(None, None, None)
                    self.image_handler = None
                    logger.info("Cleaned up image handler")
                except Exception as e:
                    logger.error(f"Error cleaning up handler: {e}")

    def __del__(self):
        """Destructeur pour s'assurer que les ressources sont nettoyées"""
        if self.image_handler:
            try:
                self.image_handler.__exit__(None, None, None)
                self.image_handler = None
            except Exception as e:
                logger.error(f"Error cleaning up in predict destructor: {e}")


def load_predictor(version):
    logger.info(f"Loading model version: {version}")

    # Configurer l'URL MLflow correcte
    mlflow.set_tracking_uri("http://mlflow-ui:5000")
    
    try:
        # Récupérer les artefacts du modèle
        client = mlflow.tracking.MlflowClient()
        
        # Récupérer l'expérience
        experiment = mlflow.get_experiment_by_name("Rakuten Model Training")
        if not experiment:
            raise ValueError("Experiment 'Rakuten Model Training' not found")
            
        logger.info(f"Found experiment ID: {experiment.experiment_id}")
        
        # Chercher le dernier run réussi
        runs = mlflow.search_runs(experiment.experiment_id)
        if runs.empty:
            raise ValueError("No runs found for the experiment")
            
        run_id = runs.iloc[0].run_id
        logger.info(f"Using run_id: {run_id}")

        # Définir le chemin des artefacts
        artifact_path = f"/app/mlruns/{experiment.experiment_id}/{run_id}/artifacts"
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact path not found at {artifact_path}")
        
        logger.info(f"Loading artifacts from: {artifact_path}")

        # Charger les artefacts
        logger.info("Loading model artifacts...")
        tokenizer_path = os.path.join(artifact_path, "tokenizer_config.json")
        best_weights_path = os.path.join(artifact_path, "best_weights.json")
        lstm_model_path = os.path.join(artifact_path, "best_lstm_model.keras")
        vgg16_model_path = os.path.join(artifact_path, "best_vgg16_model.keras")

        # Vérifier l'existence des fichiers
        for path in [tokenizer_path, best_weights_path, lstm_model_path, vgg16_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required artifact not found: {path}")

        # Charger le tokenizer
        with open(tokenizer_path, "r", encoding="utf-8") as json_file:
            tokenizer = tokenizer_from_json(json_file.read())
        logger.info("Loaded tokenizer")

        # Charger les poids
        with open(best_weights_path, "r") as json_file:
            best_weights = json.load(json_file)
        logger.info(f"Loaded best weights: {best_weights}")

        # Charger les modèles
        lstm = keras.models.load_model(lstm_model_path)
        logger.info("Loaded LSTM model")
        
        vgg16 = keras.models.load_model(vgg16_model_path)
        logger.info("Loaded VGG16 model")

        # Charger le mapper
        mapper_path = "models/mapper.json"
        if not os.path.exists(mapper_path):
            raise FileNotFoundError(f"Mapper file not found at {mapper_path}")
            
        with open(mapper_path, "r") as json_file:
            mapper = json.load(json_file)
        logger.info("Loaded category mapper")

        logger.info("All artifacts loaded successfully")
        return Predict(tokenizer, lstm, vgg16, best_weights, mapper)

    except Exception as e:
        logger.error(f"Error loading predictor: {str(e)}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Make predictions using model from MLflow")
    parser.add_argument(
        "--version",
        default=1,
        type=int,
        help="Version of the model to use"
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Run in evaluation mode (uses test data)"
    )
    args = parser.parse_args()

    try:
        # Charger le prédicteur
        predictor = load_predictor(args.version)

        # Déterminer le mode et charger les données appropriées
        if args.evaluation:
            logger.info("Running in evaluation mode with test data...")
            df = pd.DataFrame(list(sync_db.labeled_test.find()))
            # Vérifier que gridfs_file_id est présent
            if 'gridfs_file_id' not in df.columns:
                raise ValueError("gridfs_file_id column is missing in the test data")
        else:
            logger.info("Running in prediction mode with new data...")
            df = pd.DataFrame(list(sync_db.preprocessed_x_test.find())).head(10)  # Charger les données non étiquetées
            # Vérifier que 'productid' et 'imageid' sont présents
            required_columns = ['productid', 'imageid', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in test data: {missing_columns}")

        logger.info(f"Loaded {len(df)} samples")

        # Faire les prédictions
        predictions = predictor.predict(df)

        # Sauvegarder les prédictions
        output_path = "data/preprocessed/predictions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(predictions, json_file, indent=2)

        logger.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logger.error(f"Prediction process failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
