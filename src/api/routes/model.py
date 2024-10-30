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
        print("load data")
        data_importer = DataImporter()

        # Charger les données
        df = data_importer.load_data()
        _, _, X_eval, _, _, y_eval = data_importer.split_train_test(df)

        # 30% d'échantillons
        sample_size = int(len(X_eval) * 0.3)
               
        X_eval_sample = X_eval.sample(n=min(sample_size, len(X_eval)), random_state=42)
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

        # Enregistrement des métriques dans MongoDB
        evaluation_data = {
            "model_version": version,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        collection.insert_one(evaluation_data)  # Insérer l'enregistrement dans MongoDB

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

# Définir les chemins constants
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/preprocessed"
BUCKET_URL = "https://mlops-project-db.s3.eu-west-1.amazonaws.com/classification_e-commerce/"
FILENAMES = ["X_test_update.csv", "X_train_update.csv", "Y_train_CVw08PX.csv"]

class DataPipelineMetadata(BaseModel):
    execution_date: str
    raw_data_files: list
    processed_records: int
    features_info: dict
    status: str
    error_message: str = ""
    warnings: list = []

@router.post("/prepare-data/")
async def prepare_data():
    """
    Exécute le pipeline de préparation des données :
    1. Copie les données depuis /app/source_data vers data/raw
    2. Prépare le dataset
    3. Construit les features
    4. Stocke les métadonnées dans MongoDB
    """
    try:
        metadata = {
            "execution_date": datetime.now().isoformat(),
            "raw_data_files": FILENAMES,
            "processed_records": 0,
            "features_info": {},
            "status": "started",
            "error_message": "",
            "warnings": []
        }

        source_data_path = os.getenv("SOURCE_DATA_PATH", "/app/source_data")
        logger.info(f"Starting data pipeline from source path: {source_data_path}")

        # Vérifier que le dossier source existe
        if not os.path.exists(source_data_path):
            raise Exception(f"Source data folder not found at: {source_data_path}")

        # Créer la structure de dossiers
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        os.makedirs(os.path.join(RAW_DATA_PATH, "image_train"), exist_ok=True)
        os.makedirs(os.path.join(RAW_DATA_PATH, "image_test"), exist_ok=True)
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, "image_train"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, "image_test"), exist_ok=True)

        # Copier les fichiers CSV
        csv_files = [
            "X_test_update.csv",
            "X_train_update.csv",
            "Y_train_CVw08PX.csv"
        ]

        for csv_file in csv_files:
            source_path = os.path.join(source_data_path, csv_file)
            dest_path = os.path.join(RAW_DATA_PATH, csv_file)
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {csv_file} to raw data directory")
            else:
                raise Exception(f"CSV file not found: {source_path}")

        # Copier les dossiers d'images
        for folder in ["image_train", "image_test"]:
            source_folder = os.path.join(source_data_path, folder)
            dest_folder = os.path.join(RAW_DATA_PATH, folder)
            
            if os.path.exists(source_folder):
                # Si le dossier de destination existe déjà, le supprimer
                if os.path.exists(dest_folder):
                    shutil.rmtree(dest_folder)
                
                # Copier le dossier complet
                shutil.copytree(source_folder, dest_folder)
                n_images = len(os.listdir(dest_folder))
                logger.info(f"Copied {n_images} images to {folder}")
                metadata["warnings"].append(f"Copied {n_images} images to {folder}")
            else:
                warning_msg = f"Image folder not found: {source_folder}"
                metadata["warnings"].append(warning_msg)
                logger.warning(warning_msg)

        # 2. Make dataset
        logger.info("Processing dataset...")
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['make_dataset', str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH)]
            make_dataset_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv

        # 3. Build features
        logger.info("Building features...")
        data_importer = DataImporter(PROCESSED_DATA_PATH)
        df = data_importer.load_data()
        
        # Prétraitement
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(df, ["description"])
        
        image_preprocessor = ImagePreprocessor()
        image_preprocessor.preprocess_images_in_df(df)

        # Split des données
        X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)

        # Vérifier les fichiers traités
        raw_files = os.listdir(RAW_DATA_PATH)
        preprocessed_files = os.listdir(PROCESSED_DATA_PATH)
        
        metadata["warnings"].extend([
            f"Files in raw: {raw_files}",
            f"Files in preprocessed: {preprocessed_files}"
        ])

        # Fonction utilitaire pour convertir les clés en strings
        def convert_keys_to_str(d):
            if isinstance(d, dict):
                return {str(k): convert_keys_to_str(v) for k, v in d.items()}
            return d

        # Dans la partie mise à jour des métadonnées
        metadata.update({
            "processed_records": len(df),
            "features_info": {
                # Informations sur les features (X)
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "features": list(X_train.columns),
                
                # Informations sur les labels (Y)
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
                
                # Information sur la forme des données
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

        # Avant de sauvegarder dans MongoDB
        try:
            # Convertir toutes les clés en strings de manière récursive
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
        
@router.get("/test-drive-access/")
async def test_drive_access():
    try:
        source_data_path = os.getenv("SOURCE_DATA_PATH", "/app/source_data")
        
        # Vérifier l'existence du chemin source
        source_exists = os.path.exists(source_data_path)
        
        # Tenter de lister le contenu s'il existe
        source_contents = []
        if source_exists:
            try:
                source_contents = os.listdir(source_data_path)
            except Exception as e:
                source_contents = f"Error listing directory: {str(e)}"

        return {
            "system": platform.system(),
            "source_data_path": source_data_path,
            "source_exists": source_exists,
            "source_contents": source_contents,
            "current_directory": os.getcwd(),
            "root_contents": os.listdir("/")
        }
    except Exception as e:
        logger.error(f"Error in test_drive_access: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error testing drive access: {str(e)}"
        )
