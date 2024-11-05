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


@router.post("/load-data/")
async def load_data():
    """
    Route pour charger les données dans MongoDB.
    Cette opération peut prendre plusieurs minutes.
    """
    try:
        logger.info("Starting data loading process")
        
        # Liste des fichiers à traiter
        preprocessed_files = [
            "X_test_update.csv",
            "X_train_update.csv",
            "Y_train_CVw08PX.csv",
            "image_test",
            "image_train"
        ]

        # Vérifier que les fichiers existent
        for file in preprocessed_files:
            path = f"/app/data/preprocessed/{file}"
            if not os.path.exists(path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Required file not found: {file}"
                )

        # Lancer le chargement des données de manière asynchrone
        try:
            from src.scripts.data.mongodb_data_loader import MongoDBDataLoader
            loader = MongoDBDataLoader()
            
            # Enregistrer le début du chargement
            db["data_pipeline"].insert_one({
                "status": "loading",
                "start_time": datetime.now().isoformat(),
                "files": preprocessed_files
            })
            
            # Charger les données
            loader.load_all_data()
            
            # Mettre à jour le statut
            db["data_pipeline"].insert_one({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "files": preprocessed_files
            })
            
            return {
                "message": "Data loading completed successfully",
                "files_processed": preprocessed_files
            }
            
        except Exception as load_error:
            # Enregistrer l'erreur
            db["data_pipeline"].insert_one({
                "status": "failed",
                "error_time": datetime.now().isoformat(),
                "error_message": str(load_error),
                "files": preprocessed_files
            })
            
            raise HTTPException(
                status_code=500,
                detail=f"Error during data loading: {str(load_error)}"
            )
            
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@router.get("/data-status/")
def get_data_status():
    """
    Route pour vérifier si les données sont chargées dans MongoDB.
    """
    try:
        # Vérifier les collections de données tabulaires
        required_collections = {
            'preprocessed_x_train': 'Données d\'entraînement X',
            'preprocessed_x_test': 'Données de test X',
            'preprocessed_y_train': 'Labels d\'entraînement Y',
        }
        
        existing_collections = db.list_collection_names()
        
        # Vérifier les collections
        collections_status = {}
        total_tabular_documents = 0
        
        for collection_name, description in required_collections.items():
            count = db[collection_name].count_documents({}) if collection_name in existing_collections else 0
            collections_status[collection_name] = {
                "exists": collection_name in existing_collections,
                "count": count,
                "description": description
            }
            total_tabular_documents += count
            
        # Vérifier les images dans GridFS avec distinction test/train
        images_test_count = db.fs.files.count_documents(
            {"metadata.original_path": {"$regex": "/image_test/"}}
        )
        images_train_count = db.fs.files.count_documents(
            {"metadata.original_path": {"$regex": "/image_train/"}}
        )
        
        # Obtenir quelques exemples de métadonnées pour vérification
        example_test = db.fs.files.find_one({"metadata.original_path": {"$regex": "/image_test/"}})
        example_train = db.fs.files.find_one({"metadata.original_path": {"$regex": "/image_train/"}})
        
        image_examples = {
            "test": example_test["metadata"] if example_test else None,
            "train": example_train["metadata"] if example_train else None
        }
        
        return {
            "tabular_data": {
                "collections": collections_status,
                "total_documents": total_tabular_documents
            },
            "images": {
                "test": {
                    "count": images_test_count,
                    "example_metadata": image_examples["test"]
                },
                "train": {
                    "count": images_train_count,
                    "example_metadata": image_examples["train"]
                },
                "total_images": images_test_count + images_train_count
            },
            "storage_details": {
                "gridfs_files_collection": "fs.files",
                "gridfs_chunks_collection": "fs.chunks",
                "explanation": "Les images sont stockées dans GridFS avec leurs métadonnées d'origine préservées"
            },
            "is_ready": all(
                s["exists"] and s["count"] > 0 
                for s in collections_status.values()
            ) and (images_test_count > 0 and images_train_count > 0)
        }
        
    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check data status: {str(e)}"
        )
    
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
        # Utiliser MongoDB pour charger les données de test
        data_importer = DataImporter(use_mongodb=True)
        
        # Lire les données de test depuis MongoDB
        try:
            x_test = pd.DataFrame(list(data_importer.db.preprocessed_x_test.find({}, {'_id': 0})))[:10]
        except Exception as mongo_error:
            logger.error(f"MongoDB error: {mongo_error}")
            # Fallback vers le fichier CSV si MongoDB échoue
            file_path = "data/preprocessed/X_test_update.csv"
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Test data not found")
            x_test = pd.read_csv(file_path)[:10]

        # Charger le prédicteur
        predictor = load_predictor(version)

        # Appel de la méthode de prédiction
        try:
            predictions = predictor.predict(x_test, images_folder)
            
            if isinstance(predictions, dict) and "error" in predictions:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {predictions['error']}"
                )

            # Sauvegarder les prédictions dans MongoDB
            try:
                db["predictions"].insert_one({
                    "version": version,
                    "date": datetime.now().isoformat(),
                    "predictions": predictions
                })
            except Exception as mongo_error:
                logger.warning(f"Failed to save predictions to MongoDB: {mongo_error}")
                # Fallback vers JSON si MongoDB échoue
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
        logger.info(f"Starting model evaluation for version {version}")
        
        try:
            # Chargement et préparation des données avec MongoDB
            logger.info("Loading and preparing data...")
            data_importer = DataImporter(use_mongodb=True)
            
            # Utiliser le chargement par chunks pour les gros fichiers
            df = data_importer.load_data()
            _, _, X_eval, _, _, y_eval = data_importer.split_train_test(df)
            
            # Réduire l'échantillon initial et utiliser une stratification
            sample_size = min(int(len(X_eval) * 0.3), 1000)  # Max 1000 échantillons
            
            # Échantillonnage stratifié pour maintenir la distribution des classes
            X_eval_sample = X_eval.groupby(y_eval, group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), int(sample_size/len(y_eval.unique()))), 
                                 random_state=42)
            ).reset_index(drop=True)
            
            y_eval_sample = y_eval.loc[X_eval_sample.index]
            
            # Nettoyage mémoire immédiat et libération des ressources
            del df, X_eval, y_eval
            import gc
            gc.collect()
            
            logger.info(f"Prepared evaluation dataset with {len(X_eval_sample)} samples")
            
        except Exception as data_error:
            logger.error(f"Data loading error: {data_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error loading data: {str(data_error)}"
            )

        try:
            # 3. Prédictions optimisées avec traitement par lots
            logger.info("Starting predictions...")
            predictor = load_predictor(version)
            start_time = time.time()
            
            # Traitement par lots pour éviter la surcharge mémoire
            BATCH_SIZE = 32
            predictions = {}
            total_batches = len(X_eval_sample) // BATCH_SIZE + (1 if len(X_eval_sample) % BATCH_SIZE else 0)
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(X_eval_sample))
                
                batch = X_eval_sample.iloc[start_idx:end_idx]
                batch_predictions = predictor.predict(batch, image_path)
                predictions.update(batch_predictions)
                
                # Log de progression
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                
                # Nettoyage après chaque lot
                del batch_predictions
                gc.collect()
            
            inference_time = (time.time() - start_time) * 1000
            mean_inference_time = inference_time / len(X_eval_sample)
            
            logger.info(f"Predictions completed in {inference_time:.2f}ms")
            
        except Exception as pred_error:
            logger.error(f"Prediction error: {pred_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(pred_error)}"
            )

        try:
            # 4. Calcul des métriques optimisé
            logger.info("Computing metrics...")
            
            # Vectoriser la conversion des prédictions
            y_pred = np.zeros(len(X_eval_sample), dtype=int)
            for i, pred in predictions.items():
                y_pred[int(i)] = int(pred)
            
            # Vérification de la cohérence des données
            if len(y_pred) != len(y_eval_sample):
                raise ValueError(f"Prediction length mismatch: {len(y_pred)} vs {len(y_eval_sample)}")
                
            metrics = {
                "precision": float(precision_score(y_eval_sample, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_eval_sample, y_pred, average="macro", zero_division=0)),
                "f1_score": float(f1_score(y_eval_sample, y_pred, average="macro", zero_division=0))
            }
            
            # Ajouter des métriques par classe
            class_metrics = {
                f"class_{label}_f1": score 
                for label, score in zip(
                    y_eval_sample.unique(),
                    f1_score(y_eval_sample, y_pred, average=None, zero_division=0)
                )
            }
            metrics.update(class_metrics)
            
        except Exception as metric_error:
            logger.error(f"Metrics calculation error: {metric_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating metrics: {str(metric_error)}"
            )

        try:
            # 5. Sauvegarde MongoDB avec retry
            from pymongo.errors import PyMongoError
            from tenacity import retry, stop_after_attempt, wait_exponential
            
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
            def save_to_mongodb(data):
                collection.insert_one(data)
            
            evaluation_data = {
                "model_version": version,
                "evaluation_date": datetime.now().isoformat(),
                "metrics": metrics,
                "inference_performance": {
                    "mean_inference_time_ms": float(mean_inference_time),
                    "total_inference_time_ms": float(inference_time),
                    "sample_size": len(X_eval_sample),
                    "batch_size": BATCH_SIZE
                }
            }
            
            save_to_mongodb(evaluation_data)
            logger.info("Results saved to MongoDB")
            
        except Exception as db_error:
            logger.error(f"Database error: {db_error}", exc_info=True)
            logger.warning("Failed to save results to database, but evaluation completed")

        logger.info("Model evaluation completed successfully")
        return {
            "metrics": metrics,
            "mean_inference_time_ms": float(mean_inference_time)
        }

    except Exception as e:
        logger.error(f"Unexpected error in evaluate_model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

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