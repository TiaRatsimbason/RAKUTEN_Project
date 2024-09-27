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
from scripts.features.build_features import DataImporter

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
async def evaluate_model():
    try:
        print("load data")
        data_importer = DataImporter()

        # Charger les données
        df = data_importer.load_data()
        _, _, X_eval, _, _, y_eval = data_importer.split_train_test(df)
        
        # Réduction de la taille des données de test à utiliser pour l'evaluation du modèle (ici 10% des données) --> en utilisant toutes les données de test le conteneur crash
        X_eval_sample = X_eval.sample(frac=0.1, random_state=42)
        y_eval_sample = y_eval.loc[X_eval_sample.index]

        # Charger le prédicteur au démarrage de l'application
        predictor = load_predictor()

        print("prediction")
        # Prédictions avec l'échantillon réduit
        predictions = predictor.predict(X_eval_sample, "data/preprocessed/image_train")

        print("Mapping")
        with open("models/mapper.json", "r") as json_file:
            mapper = json.load(json_file)

        # Mapping des vraies valeurs avec l'échantillon réduit
        mapped_y_eval = []
        for val in y_eval_sample.values.flatten():
            mapped_y_eval.append(mapper[f"{val}"])

        print("Mapping predictions")
        mapped_predictions = [str(pred) for pred in predictions.values()]

        print("Calcul score")
        # Calcul des métriques avec l'échantillon réduit
        precision = precision_score(
            mapped_y_eval, mapped_predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            mapped_y_eval, mapped_predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            mapped_y_eval, mapped_predictions, average="macro", zero_division=0
        )

        # Retour des résultats d'évaluation
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

