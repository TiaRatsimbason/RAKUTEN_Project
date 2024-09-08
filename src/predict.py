from flask import Blueprint, request, jsonify
from src.features.build_features import TextPreprocessor, ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd

# Définition du blueprint pour les prédictions
predict_blueprint = Blueprint('predict', __name__)

class Predict:
    def __init__(self, tokenizer, lstm, vgg16, best_weights, mapper):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, df, image_path_base):
        # Prétraitement des descriptions textuelles
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(df, columns=["description"])

        # Prétraitement des images
        image_preprocessor = ImagePreprocessor(image_path_base)
        image_preprocessor.preprocess_images_in_df(df)

        # Convertir le texte en séquences de tokens
        sequences = self.tokenizer.texts_to_sequences(df["description"])
        padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")

        # Prétraiter les images
        target_size = (224, 224, 3)
        images = df["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        # Faire les prédictions avec les modèles LSTM et VGG16
        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        # Combiner les probabilités selon les poids
        concatenate_proba = (self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba)
        final_predictions = np.argmax(concatenate_proba, axis=1)

        # Mapper les résultats aux catégories
        predictions = {i: self.mapper[str(final_predictions[i])] for i in range(len(final_predictions))}

        return predictions

def load_predictor():
    # Charger les configurations et modèles
    with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    lstm = keras.models.load_model("models/best_lstm_model.h5")
    vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

    with open("models/best_weights.json", "r") as json_file:
        best_weights = json.load(json_file)

    with open("models/mapper.json", "r") as json_file:
        mapper = json.load(json_file)

    return Predict(tokenizer, lstm, vgg16, best_weights, mapper)

# Route pour l'API de prédiction
@predict_blueprint.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Charger le prédicteur
        predictor = load_predictor()

        # Charger le fichier de données fourni par la requête
        file = request.files['file']
        df = pd.read_csv(file)

        # Chemin de base des images (à adapter selon la structure du projet)
        image_path_base = 'data/preprocessed/image_train'

        # Faire la prédiction
        predictions = predictor.predict(df, image_path_base)

        # Renvoyer les prédictions en JSON
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
