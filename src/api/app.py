from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow import keras

app = FastAPI()

# Charger les configurations et modèles
with open("src/models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

lstm = load_model("src/models/best_lstm_model.h5")
vgg16 = load_model("src/models/best_vgg16_model.h5")

with open("src/models/best_weights.json", "r") as json_file:
    best_weights = json.load(json_file)

with open("src/models/mapper.json", "r") as json_file:
    mapper = json.load(json_file)

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...), images_folder: str = "data/preprocessed/image_test"):
    # Lire le fichier CSV envoyé et le convertir en DataFrame
    df = pd.read_csv(file.file)
    
    # Prétraitement des descriptions textuelles
    sequences = tokenizer.texts_to_sequences(df["description"])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
    
    # Prétraitement des images
    target_size = (224, 224, 3)
    images = df["image_path"].apply(lambda x: preprocess_image(f"{images_folder}/{x}", target_size))
    images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)
    
    # Prédictions avec les modèles
    lstm_proba = lstm.predict([padded_sequences])
    vgg16_proba = vgg16.predict([images])
    
    # Combinaison des probabilités
    concatenate_proba = (best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba)
    final_predictions = np.argmax(concatenate_proba, axis=1)
    
    # Mapper les résultats aux catégories
    predictions = {i: mapper[str(final_predictions[i])] for i in range(len(final_predictions))}
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

