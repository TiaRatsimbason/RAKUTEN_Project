# Standard library imports
import json
import os
import shutil
import subprocess

# Third-party library imports
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

# Local/application-specific imports
from scripts.predict import load_predictor

router = APIRouter()

# Charger le prédicteur au démarrage de l'application
predictor = load_predictor()


@router.post("/train-model/")
async def train_model():
    try:
        # Execute the main.py script to train the model
        subprocess.run(["python", "src/scripts/main.py"], check=True)
        return {"message": "Model training completed successfully."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in training model: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during model training: {e}"
        )


@router.post("/predict/")
async def predict(
    file: UploadFile = File(...), images_folder: str = "data/preprocessed/image_test"
):
    try:
        # Sauvegarder le fichier temporairement
        with open("temp.csv", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Lire le fichier CSV et le convertir en DataFrame
        df = pd.read_csv("temp.csv")[:10]

        # Appel de la méthode de prédiction
        predictions = predictor.predict(df, images_folder)

        # Sauvegarder les prédictions dans un fichier JSON dans le répertoire "data/preprocessed"
        output_path = "data/preprocessed/predictions.json"
        with open(output_path, "w") as json_file:
            json.dump(predictions, json_file, indent=2)

        # Supprimer le fichier temporaire après utilisation
        os.remove("temp.csv")

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {e}"
        )
