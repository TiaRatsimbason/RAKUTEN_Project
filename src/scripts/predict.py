import argparse
import json
import logging
import os

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

from src.scripts.features.build_features import TextPreprocessor, ImagePreprocessor
from src.scripts.data.gridfs_image_handler import GridFSImageHandler

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

    def predict(self, df, image_path_base=None):
        """
        Fait des prédictions en utilisant les images depuis GridFS ou le système de fichiers
        """
        # Prétraitement du texte
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(df, columns=["description"])

        # Préparer les séquences de texte
        sequences = self.tokenizer.texts_to_sequences(df["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        # Traitement des images
        if image_path_base is None:  # Utiliser GridFS
            with GridFSImageHandler() as handler:
                logger.info("Loading images from GridFS...")
                images = []
                for _, row in df.iterrows():
                    img_path = handler.get_image_path(row['imageid'], row['productid'])
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    images.append(img_array)
                images = np.array(images)
        else:  # Utiliser le système de fichiers
            logger.info("Loading images from filesystem...")
            image_preprocessor = ImagePreprocessor(image_path_base)
            image_preprocessor.preprocess_images_in_df(df)
            images = df["image_path"].apply(
                lambda x: self.preprocess_image(x, (224, 224, 3))
            )
            images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        # Faire les prédictions
        logger.info("Making predictions with LSTM model...")
        lstm_proba = self.lstm.predict([padded_sequences])
        
        logger.info("Making predictions with VGG16 model...")
        vgg16_proba = self.vgg16.predict([images])

        # Combiner les probabilités selon les poids         
        logger.info("Combining predictions...")
        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)
        
        # Mapper les résultats aux catégories
        logger.info("Mapping predictions to categories...")
        inverse_mapper = {str(v): k for k, v in self.mapper.items()}
        
        predictions = {}
        for i in range(len(final_predictions)):
            pred_value = str(final_predictions[i])
            if pred_value in inverse_mapper:
                predictions[str(i)] = inverse_mapper[pred_value]
            else:
                logger.warning(f"Value {pred_value} not found in inverse mapper")

        return predictions

    def preprocess_image(self, image_path, target_size):
        """Méthode utilitaire pour le prétraitement des images depuis le système de fichiers"""
        img = load_img(image_path, target_size=target_size[:2])
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array


def load_predictor(version):
    logger.info(f"Loading model version: {version}")

    # Récupérer les artefacts du modèle (tokenizer, mapper, etc.)
    client = mlflow.tracking.MlflowClient()
    logger.info(f"Getting run_id for model version: {version}")
    model_version_details = client.get_model_version(
        name="RegisteredConcatenateModel", version=str(version)
    )
    run_id = model_version_details.run_id
    logger.info(f"Obtained run_id: {run_id}")

    # Utiliser le run_id pour récupérer le détail du run et obtenir l'ID de l'expérience
    run_details = client.get_run(run_id)
    experiment_id = run_details.info.experiment_id
    logger.info(f"Obtained experiment_id: {experiment_id}")

    # Définir le chemin des artefacts
    artifact_path = f"/app/mlruns/{experiment_id}/{run_id}/artifacts"
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact path not found at {artifact_path}")

    logger.info(f"Downloading individual artifacts from {artifact_path}")
    
    # Télécharger chaque artefact individuellement
    tokenizer_path = os.path.join(artifact_path, "tokenizer_config.json")
    best_weights_path = os.path.join(artifact_path, "best_weights.json")
    lstm_model_path = os.path.join(artifact_path, "best_lstm_model.keras")
    vgg16_model_path = os.path.join(artifact_path, "best_vgg16_model.keras")
    
    logger.info("Artifacts paths set successfully")
    
    # Charger le tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer config file not found at {tokenizer_path}")
    logger.info("Loading tokenizer from downloaded artifacts")
    with open(tokenizer_path, "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

    # Charger les poids optimaux depuis les artefacts
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Best weights file not found at {best_weights_path}")
    logger.info("Loading best weights from downloaded artifacts")
    with open(best_weights_path, "r") as json_file:
        best_weights = json.load(json_file)

    # Charger les modèles lstm et vgg16
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}")
    logger.info("Loading LSTM model from downloaded artifacts")
    lstm = keras.models.load_model(lstm_model_path)

    if not os.path.exists(vgg16_model_path):
        raise FileNotFoundError(f"VGG16 model not found at {vgg16_model_path}")
    logger.info("Loading VGG16 model from downloaded artifacts")
    vgg16 = keras.models.load_model(vgg16_model_path)

    # Charger le mapper
    with open("models/mapper.json", "r") as json_file:
        mapper = json.load(json_file)
    logger.info(f"Loaded mapper: {mapper}")

    # Retourner l'instance de Predict avec tous les paramètres requis
    return Predict(tokenizer, lstm, vgg16, best_weights, mapper)


def main():
    parser = argparse.ArgumentParser(description="Input data")

    parser.add_argument(
        "--dataset_path",
        default="data/preprocessed/X_train_update.csv",
        type=str,
        help="File path for the input CSV file.",
    )
    parser.add_argument(
        "--images_path",
        default=None,
        type=str,
        help="Base path for the images. If not provided, will use GridFS.",
    )
    parser.add_argument(
        "--version",
        default=1,
        type=int,
        help="Version of the model to use.",
    )
    parser.add_argument(
        "--use_gridfs",
        action='store_true',
        help="Use GridFS instead of filesystem for images."
    )
    args = parser.parse_args()

    # Charger le prédicteur
    predictor = load_predictor(args.version)

    # Charger les données
    if args.dataset_path.endswith('.csv'):
        df = pd.read_csv(args.dataset_path)
    else:
        # Charger depuis MongoDB si ce n'est pas un CSV
        from pymongo import MongoClient
        client = MongoClient("mongodb://admin:motdepasseadmin@mongo:27017/")
        db = client['rakuten_db']
        df = pd.DataFrame(list(db.preprocessed_x_test.find({}, {'_id': 0})))

    # Faire les prédictions
    image_path = None if args.use_gridfs else args.images_path
    predictions = predictor.predict(df, image_path)

    # Sauvegarder les prédictions
    output_path = "data/preprocessed/predictions.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, indent=2)

    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()