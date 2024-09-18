# Standard library imports
import json
import os
import shutil
import subprocess

# Third-party library imports
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sklearn.metrics import precision_score, recall_score, f1_score

# Local/application-specific imports
from scripts.predict import load_predictor

router = APIRouter()


@router.post("/train-model/")
async def train_model():
    try:
        # Execute the main.py script to train the model
        subprocess.run(["python", "scripts/main.py"], check=True)
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

        # Charger le prédicteur au démarrage de l'application
        predictor = load_predictor()

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


@router.post("/evaluate-model/")
async def evaluate_model(n_samples: int = 1000):
    try:
        X_eval = pd.read_csv("data/preprocessed/X_eval_update.csv").sample(
            n=n_samples, random_state=42
        )
        y_eval = pd.read_csv("data/preprocessed/Y_eval_update.csv").sample(
            n=n_samples, random_state=42
        )

        if y_eval.shape[1] > 1:
            y_eval = y_eval.iloc[:, 1]

        # Charger le prédicteur au démarrage de l'application
        predictor = load_predictor()

        predictions = predictor.predict(X_eval, "data/preprocessed/image_eval")

        with open("models/mapper.json", "r") as json_file:
            mapper = json.load(json_file)

        mapped_predictions = [mapper[str(pred)] for pred in predictions]

        y_eval = y_eval.astype(str)
        mapped_predictions = [str(pred) for pred in mapped_predictions]

        precision = precision_score(
            y_eval, mapped_predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_eval, mapped_predictions, average="macro", zero_division=0
        )
        f1 = f1_score(y_eval, mapped_predictions, average="macro", zero_division=0)

        return {
            "evaluation_report": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during model evaluation: {e}"
        )
