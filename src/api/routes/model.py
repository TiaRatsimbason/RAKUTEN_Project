# Standard library imports
import json
import os
import subprocess
import logging
from fastapi import HTTPException, Query
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime
from src.scripts.data.import_raw_data import import_raw_data
from src.scripts.data.make_dataset import main as make_dataset
from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.scripts.data.check_structure import check_existing_folder, check_existing_file
import sys
from src.scripts.data.make_dataset import main as make_dataset_main
import shutil
from src.scripts.data.setup_data_cloud import create_directory_structure, copy_files_and_folders_from_drive
import platform
import time
import numpy as np

# Configuration MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:motdepasseadmin@mongo:27017/")
client = MongoClient(MONGODB_URI)
db = client["rakuten_db"]  # Nom de la base de données
collection = db["model_evaluation"]  # Nom de la collection pour les évaluations

# Configuration du logger
logging.basicConfig(level=logging.INFO)  
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
        # 1. Charger et préparer les données de manière minimale
        data_importer = DataImporter()
        df = data_importer.load_data()
        _, _, X_eval, _, _, y_eval = data_importer.split_train_test(df)
        
        # Échantillonnage 30%
        sample_size = int(len(X_eval) * 0.3)
        X_eval_sample = X_eval.sample(n=min(sample_size, len(X_eval)), random_state=42)
        y_eval_sample = y_eval.loc[X_eval_sample.index]
        
        # 2. Prédictions avec mesure du temps
        predictor = load_predictor(version)
        start_time = time.time()
        predictions = predictor.predict(X_eval_sample, "data/preprocessed/image_train")
        inference_time = (time.time() - start_time) * 1000
        mean_inference_time = inference_time / len(X_eval_sample)
        
        # 3. Calcul des métriques
        mapped_predictions = [int(pred) for pred in predictions.values()]
        metrics = {
            "precision": precision_score(y_eval_sample, mapped_predictions, average="macro", zero_division=0),
            "recall": recall_score(y_eval_sample, mapped_predictions, average="macro", zero_division=0),
            "f1_score": f1_score(y_eval_sample, mapped_predictions, average="macro", zero_division=0)
        }
        
        # 4. Sauvegarder dans MongoDB
        collection.insert_one({
            "model_version": version,
            "evaluation_date": datetime.now().isoformat(),
            "metrics": metrics,
            "inference_performance": {
                "mean_inference_time_ms": mean_inference_time,
                "total_inference_time_ms": inference_time,
                "sample_size": len(X_eval_sample)
            }
        })
        
        # 5. Retourner uniquement les résultats essentiels
        return {
            "metrics": metrics,
            "mean_inference_time_ms": mean_inference_time
        }
        
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DataPipelineMetadata(BaseModel):
    execution_date: str
    raw_data_files: list = []  # Rendre optionnel avec une liste vide par défaut
    processed_records: int
    features_info: dict
    status: str
    error_message: str = ""
    warnings: list = []

@router.post("/prepare-data/")
async def prepare_data():
    """
    Exécute le pipeline de préparation des données :
    1. Charge les données prétraitées
    2. Construit les features
    3. Stocke les métadonnées dans MongoDB
    """
    try:
        # Liste des fichiers prétraités
        preprocessed_files = [
            "X_test_update.csv",
            "X_train_update.csv",
            "Y_train_CVw08PX.csv",
            "image_test",
            "image_train"
        ]

        metadata = {
            "execution_date": datetime.now().isoformat(),
            "raw_data_files": preprocessed_files,  # Ajouter la liste des fichiers
            "processed_records": 0,
            "features_info": {},
            "status": "started",
            "error_message": "",
            "warnings": []
        }

        logger.info("Loading preprocessed data...")

        # Charger directement les données prétraitées
        data_importer = DataImporter("/app/data/preprocessed")
        df = data_importer.load_data()
        
        # Prétraitement
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(df, ["description"])
        
        image_preprocessor = ImagePreprocessor()
        image_preprocessor.preprocess_images_in_df(df)

        # Split des données
        X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)

        def convert_keys_to_str(d):
            if isinstance(d, dict):
                return {str(k): convert_keys_to_str(v) for k, v in d.items()}
            return d

        # Mettre à jour les métadonnées
        metadata.update({
            "processed_records": len(df),
            "features_info": {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "features": list(X_train.columns),
                
                "labels_info": {
                    "train_labels": {
                        "count": len(y_train),
                        "unique_classes": len(y_train.unique()),
                        "class_distribution": convert_keys_to_str(y_train.value_counts().to_dict())
                    },
                    "val_labels": {
                        "count": len(y_val),
                        "unique_classes": len(y_val.unique()),
                        "class_distribution": convert_keys_to_str(y_val.value_counts().to_dict())
                    },
                    "test_labels": {
                        "count": len(y_test),
                        "unique_classes": len(y_test.unique()),
                        "class_distribution": convert_keys_to_str(y_test.value_counts().to_dict())
                    }
                },
                
                "data_split_info": {
                    "train": {
                        "X_shape": list(X_train.shape),
                        "y_shape": list(y_train.shape)
                    },
                    "val": {
                        "X_shape": list(X_val.shape),
                        "y_shape": list(y_val.shape)
                    },
                    "test": {
                        "X_shape": list(X_test.shape),
                        "y_shape": list(y_test.shape)
                    }
                }
            },
            "status": "completed"
        })

        # Sauvegarder dans MongoDB
        try:
            metadata_safe = convert_keys_to_str(metadata)
            pipeline_metadata = DataPipelineMetadata(**metadata_safe)
            db["data_pipeline"].insert_one(pipeline_metadata.dict())
        except Exception as mongo_error:
            logger.error(f"Failed to save metadata to MongoDB: {str(mongo_error)}")
            raise

        logger.info("Data pipeline completed successfully")
        return {
            "message": "Data pipeline completed successfully",
            "metadata": pipeline_metadata.dict()
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"Pipeline error: {error_message}")
        metadata.update({
            "status": "failed",
            "error_message": error_message
        })
        
        try:
            pipeline_metadata = DataPipelineMetadata(**metadata)
            db["data_pipeline"].insert_one(pipeline_metadata.dict())
        except Exception as mongo_error:
            logger.error(f"Failed to save metadata to MongoDB: {str(mongo_error)}")

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during data pipeline execution: {error_message}"
        )
