# Standard library imports
import json
import os
import subprocess
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)  # Configurer le niveau de log selon votre besoin
logger = logging.getLogger(__name__)  # Initialiser le logger avec le nom du module

# Third-party library imports
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import precision_score, recall_score, f1_score

from src.scripts.features.build_features import DataImporter
from src.scripts.predict import load_predictor

router = APIRouter()


@router.post("/train-model/")
async def train_model():
    try:
        from src.scripts.main import train_and_save_model
        train_and_save_model()
        return {"message": "Model training completed successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during model training: {e}"
        )


@router.post("/predict/")
async def predict(
    images_folder: str = Query("data/preprocessed/image_test", description="Path to the folder containing images"),
    version: int = Query(1, description="Version number of the model to use")
):
    try:
        # Lire le fichier CSV directement depuis le chemin spécifié dans le conteneur
        file_path = "data/preprocessed/X_test_update.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="CSV file not found")

        # Lire le fichier CSV et le convertir en DataFrame
        df = pd.read_csv(file_path)[:10]

        # Charger le prédicteur en utilisant la version spécifiée
        predictor = load_predictor(version)

        # Appel de la méthode de prédiction
        try:
            predictions = predictor.predict(df, images_folder)
            
            if isinstance(predictions, dict) and "error" in predictions:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {predictions['error']}"
                )

            # Sauvegarder les prédictions dans un fichier JSON
            output_path = "data/preprocessed/predictions.json"
            with open(output_path, "w") as json_file:
                json.dump(predictions, json_file, indent=2)

            return {"predictions": predictions}
        
        except Exception as pred_error:
            logger.error(f"Prediction error: {pred_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction process: {str(pred_error)}"
            )

    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


@router.post("/evaluate-model/")
async def evaluate_model(version: int = Query(1, description="Version number of the model to use")):
    try:
        print("load data")
        data_importer = DataImporter()

        # Charger les données
        df = data_importer.load_data()
        _, _, X_eval, _, _, y_eval = data_importer.split_train_test(df)

        # Réduction de la taille des données de test à utiliser pour l'evaluation du modèle (ici 10% des données)
        
        X_eval_sample = X_eval.sample(n=min(10, len(X_eval)), random_state=42)
        y_eval_sample = y_eval.loc[X_eval_sample.index]

        print(f"X_eval_sample shape: {X_eval_sample.shape}")
        logger.info(f"X_eval_sample before prediction: {X_eval_sample}")
        print(f"y_eval_sample shape: {y_eval_sample.shape}")
        logger.info(f"y_eval_sample before prediction: {y_eval_sample}")

        # Charger le prédicteur au démarrage de l'application
        predictor = load_predictor(version)

        print("prediction")
        # Prédictions avec l'échantillon réduit
        predictions = predictor.predict(X_eval_sample, "data/preprocessed/image_train")

        print(f"predictions type: {type(predictions)}")
        print(f"predictions: {predictions}")

        # Convertir les prédictions en liste
        mapped_predictions = [int(pred) for pred in predictions.values()]
        
        

        # Afficher seulement les 10 premiers échantillons pour comparaison
        print("\nComparaison des 10 premiers échantillons:")
        print(f"y_eval_sample (vraies valeurs): {y_eval_sample.values.flatten()[:10].tolist()}")
        print(f"mapped_predictions (prédictions): {mapped_predictions[:10]}")

        print("\nCalcul score")
        # Vérifier que y_true et y_pred ont le même nombre d'échantillons
        if len(y_eval_sample) != len(mapped_predictions):
            raise ValueError(f"Inconsistent number of samples: y_true has {len(y_eval_sample)} samples, y_pred has {len(mapped_predictions)} samples.")

        # Calcul des métriques avec l'échantillon réduit
        precision = precision_score(
            y_eval_sample, mapped_predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_eval_sample, mapped_predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_eval_sample, mapped_predictions, average="macro", zero_division=0
        )

        print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

        # Retour des résultats d'évaluation
        return {
            "evaluation_report": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
            }
        }

    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during model evaluation: {e}"
        )