import json
import os
import gc
import subprocess
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import uvloop
import mlflow
from fastapi import HTTPException, Query, APIRouter
from pymongo import MongoClient
from pydantic import BaseModel
from bson import ObjectId
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime
import sys
import shutil
import platform
import time
from src.scripts.data.import_raw_data import import_raw_data
from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.scripts.data.check_structure import check_existing_folder, check_existing_file
from src.scripts.data.setup_data_cloud import create_directory_structure, copy_files_and_folders_from_drive
from src.config.mongodb import async_db, async_fs, sync_db, sync_client, sync_fs
import nltk
import ssl
from src.scripts.predict import load_predictor
from pymongo.errors import PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential
from gridfs import GridFS

# Configuration optimale d'asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configuration SSL pour NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constantes
BATCH_SIZE = 1000
MAX_WORKERS = os.cpu_count()
CHUNK_SIZE = 5000
NUM_CLASSES = 27

class DataPipelineMetadata(BaseModel):
    """Modèle Pydantic pour les métadonnées du pipeline"""
    execution_date: str
    raw_data_files: list = []
    processed_records: int
    features_info: dict
    status: str
    error_message: str = ""
    warnings: list = []

class PreprocessingProgress:
    """Classe pour suivre la progression du prétraitement"""
    def __init__(self):
        self.total = 0
        self.current = 0
        self.msg = ""

    def update(self, amount, msg=""):
        self.current += amount
        if msg:
            self.msg = msg
        logger.info(f"Progress: {self.current}/{self.total} - {msg}")

progress = PreprocessingProgress()

def initialize_nltk():
    """Initialise NLTK une seule fois pour tous les workers"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def process_text_chunk(chunk_data):
    """
    Traite un chunk de texte en utilisant le préprocesseur.
    Pour être utilisé avec ProcessPoolExecutor.
    """
    try:
        initialize_nltk()
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(chunk_data, ["description"])
        return chunk_data
    except Exception as e:
        logger.error(f"Error in process_text_chunk: {str(e)}")
        raise

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
            sync_db["data_pipeline"].insert_one({
                "status": "loading",
                "start_time": datetime.now().isoformat(),
                "files": preprocessed_files
            })
            
            # Charger les données
            loader.load_all_data()
            
            # Mettre à jour le statut
            sync_db["data_pipeline"].insert_one({
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
            sync_db["data_pipeline"].insert_one({
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
async def get_data_status():
    try:
        required_collections = {
            'preprocessed_x_train': 'Données d\'entraînement X',
            'preprocessed_x_test': 'Données de test X',
            'preprocessed_y_train': 'Labels d\'entraînement Y',
        }
        
        # Utiliser le client asynchrone
        existing_collections = await async_db.list_collection_names()
        
        collections_status = {}
        total_tabular_documents = 0
        
        for collection_name, description in required_collections.items():
            count = await async_db[collection_name].count_documents({}) if collection_name in existing_collections else 0
            collections_status[collection_name] = {
                "exists": collection_name in existing_collections,
                "count": count,
                "description": description
            }
            total_tabular_documents += count
            
        # Vérifier les images avec le client asynchrone
        images_test_count = await async_db.fs.files.count_documents(
            {"metadata.original_path": {"$regex": "/image_test/"}}
        )
        images_train_count = await async_db.fs.files.count_documents(
            {"metadata.original_path": {"$regex": "/image_train/"}}
        )
        
        # Obtenir quelques exemples de métadonnées pour vérification
        # Utiliser le client synchrone pour GridFS
        fs = GridFS(sync_db)
        example_test_file = fs.find_one({"metadata.original_path": {"$regex": "/image_test/"}})
        example_train_file = fs.find_one({"metadata.original_path": {"$regex": "/image_train/"}})
        
        image_examples = {
            "test": example_test_file.metadata if example_test_file else None,
            "train": example_train_file.metadata if example_train_file else None
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
        
@router.post("/prepare-data/")
async def prepare_data():
    """
    Pipeline complet de préparation des données avec gestion des images :
    1. Charge les données et images depuis MongoDB/GridFS
    2. Prétraite texte et images
    3. Split les données
    4. Réorganise les images dans GridFS selon les splits
    5. Stocke les données labellisées avec références aux images
    """
    def clean_metadata(data):
        """Convertit les ObjectId en strings dans le dictionnaire metadata"""
        if isinstance(data, dict):
            return {
                key: clean_metadata(value) if isinstance(value, (dict, list)) else str(value) 
                if hasattr(value, '__str__') and not isinstance(value, (int, float, bool, str)) 
                else value
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [clean_metadata(item) for item in data]
        elif hasattr(data, '__str__') and not isinstance(data, (int, float, bool, str)):
            return str(data)
        return data
    try:
        logger.info("Starting optimized data preparation pipeline...")
        
        metadata = {
            "execution_date": datetime.now().isoformat(),
            "status": "started",
            "error_message": "",
            "warnings": []
        }

        # Fonction ajoutée pour vérifier les métadonnées des images
        async def check_image_metadata():
            train_count = await async_db.fs.files.count_documents({"metadata.split": "train"})
            val_count = await async_db.fs.files.count_documents({"metadata.split": "validation"})
            test_count = await async_db.fs.files.count_documents({"metadata.split": "test"})
            
            logger.info(f"Images in GridFS - Train: {train_count}, Val: {val_count}, Test: {test_count}")
            
            return train_count, val_count, test_count

        # Initialiser NLTK au démarrage
        initialize_nltk()

        try:
            # 1. Chargement optimisé des données
            logger.info("Loading initial data...")
            data_importer = DataImporter()
            df = data_importer.load_data()
            total_records = len(df)
            progress.total = total_records
            logger.info(f"Loaded {total_records} records")

            # 2. Prétraitement parallèle du texte avec gestion d'erreurs
            logger.info("Starting parallel text preprocessing...")
            chunks = np.array_split(df, MAX_WORKERS)
            
            async def process_all_chunks():
                try:
                    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        loop = asyncio.get_event_loop()
                        tasks = [
                            loop.run_in_executor(executor, process_text_chunk, chunk)
                            for chunk in chunks
                        ]
                        processed_chunks = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Vérifier les erreurs
                        errors = [chunk for chunk in processed_chunks if isinstance(chunk, Exception)]
                        if errors:
                            raise Exception(f"Errors in text preprocessing: {errors}")
                        
                        return pd.concat([
                            chunk for chunk in processed_chunks 
                            if isinstance(chunk, pd.DataFrame)
                        ], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error in process_all_chunks: {str(e)}")
                    raise

            # Traitement du texte
            try:
                df = await process_all_chunks()
                logger.info("Text preprocessing completed successfully")
            except Exception as text_error:
                logger.error(f"Text preprocessing failed: {str(text_error)}")
                raise

            # 3. Split des données
            logger.info("Performing optimized data split...")
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = await asyncio.to_thread(
                    data_importer.split_train_test, df
                )
                logger.info(f"Split completed - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            except Exception as split_error:
                logger.error(f"Data split failed: {str(split_error)}")
                raise

            # Création des DataFrames avec labels
            train_data = X_train.assign(label=y_train.values)
            val_data = X_val.assign(label=y_val.values)
            test_data = X_test.assign(label=y_test.values)

            # 4. Traitement des images
            async def process_images_batch(batch_df, split_name):
                """Traite un batch d'images avec gestion optimisée de la mémoire"""
                image_mappings = {}
                
                # Créer les critères de recherche pour MongoDB
                search_criteria = {
                    "$or": []
                }
                
                # Construire les critères de recherche
                for _, row in batch_df.iterrows():
                    search_criteria["$or"].append({
                        "metadata.productid": str(row['productid']),
                        "metadata.imageid": str(row['imageid'])
                    })
                
                try:
                    # Recherche optimisée dans GridFS avec logging
                    logger.info(f"Searching for {len(search_criteria['$or'])} images in GridFS for {split_name}")
                    cursor = async_db.fs.files.find(search_criteria)

                    async for file_doc in cursor:
                        try:
                            product_id = str(file_doc["metadata"]["productid"])
                            image_id = str(file_doc["metadata"]["imageid"])
                            key = f"{product_id}_{image_id}"
                            
                            matching_row = batch_df[
                                (batch_df['productid'].astype(str) == product_id) &
                                (batch_df['imageid'].astype(str) == image_id)
                            ]
                            
                            if not matching_row.empty:
                                new_metadata = {
                                    **file_doc["metadata"],
                                    "split": split_name,
                                    "label": int(matching_row['label'].iloc[0]),
                                    "original_file_id": str(file_doc["_id"])
                                }
                                
                                # Log pour le debugging
                                logger.debug(f"Processing image {key} for {split_name}")
                                
                                chunks_cursor = async_db.fs.chunks.find(
                                    {"files_id": file_doc["_id"]}
                                ).sort("n", 1)
                                
                                chunks_data = []
                                async for chunk in chunks_cursor:
                                    chunks_data.append(chunk["data"])
                                
                                if chunks_data:
                                    # Utiliser upload_from_stream au lieu de put
                                    data = b"".join(chunks_data)
                                    new_file_id = await async_fs.upload_from_stream(
                                        filename=f"{split_name}/{key}.jpg",
                                        source=data,
                                        metadata=new_metadata
                                    )
                                    
                                    image_mappings[key] = str(new_file_id)
                                    logger.debug(f"Successfully processed image {key}")
                                else:
                                    logger.warning(f"No chunks found for image {key}")
                                
                            else:
                                logger.warning(f"No matching row found for image {key}")
                            
                        except Exception as e:
                            logger.warning(f"Error processing image {key}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error in batch processing for {split_name}: {e}")
                    raise
                
                logger.info(f"Processed {len(image_mappings)} images for {split_name}")
                return image_mappings

            async def process_split_images(split_data, split_name):
                """Traite toutes les images d'un split en parallèle par batches"""
                try:
                    all_mappings = {}
                    batches = np.array_split(split_data, max(1, len(split_data) // CHUNK_SIZE))
                    
                    for i, batch in enumerate(batches):
                        logger.info(f"Processing {split_name} images batch {i+1}/{len(batches)}")
                        batch_mappings = await process_images_batch(batch, split_name)
                        all_mappings.update(batch_mappings)
                        
                    return all_mappings
                except Exception as e:
                    logger.error(f"Error processing split {split_name}: {e}")
                    raise

            # Traitement parallèle des images avec vérification ajoutée
            logger.info("Processing images for all splits in parallel...")
            try:
                train_mappings, val_mappings, test_mappings = await asyncio.gather(
                    process_split_images(train_data, "train"),
                    process_split_images(val_data, "validation"),
                    process_split_images(test_data, "test")
                )
                
                # Vérification des mappings
                logger.info(f"Image mappings created - Train: {len(train_mappings)}, Val: {len(val_mappings)}, Test: {len(test_mappings)}")
                
                if not train_mappings and not val_mappings and not test_mappings:
                    logger.error("No images were processed successfully")
                    raise Exception("Failed to process any images")
                    
            except Exception as image_error:
                logger.error(f"Image processing failed: {str(image_error)}")
                raise
            # Création des DataFrames avec labels
            train_data = X_train.assign(label=y_train.values)
            val_data = X_val.assign(label=y_val.values)
            test_data = X_test.assign(label=y_test.values)

            # 4. Prétraitement des images
            image_preprocessor = ImagePreprocessor()
            image_preprocessor.preprocess_images_in_df(train_data)
            image_preprocessor.preprocess_images_in_df(val_data)
            image_preprocessor.preprocess_images_in_df(test_data)
            # 5. Stockage MongoDB
            async def prepare_and_insert_records(data, mappings, collection_name):
                try:
                    records = []
                    for _, row in data.iterrows():
                        key = f"{str(row['productid'])}_{str(row['imageid'])}"
                        record = row.to_dict()
                        record['gridfs_file_id'] = mappings.get(key)
                        records.append(record)
                    
                    for i in range(0, len(records), BATCH_SIZE):
                        batch = records[i:i + BATCH_SIZE]
                        await async_db[collection_name].insert_many(batch)
                        logger.info(f"Inserted {len(batch)} records in {collection_name}")
                    
                    return len(records)
                except Exception as e:
                    logger.error(f"Error in prepare_and_insert_records for {collection_name}: {e}")
                    raise

            # Stockage parallèle
            logger.info("Performing parallel MongoDB storage...")
            try:
                collections_to_drop = ['labeled_train', 'labeled_val', 'labeled_test']
                await asyncio.gather(*[
                    async_db[col].drop() 
                    for col in collections_to_drop 
                    if col in await async_db.list_collection_names()
                ])
                
                train_count, val_count, test_count = await asyncio.gather(
                    prepare_and_insert_records(train_data, train_mappings, 'labeled_train'),
                    prepare_and_insert_records(val_data, val_mappings, 'labeled_val'),
                    prepare_and_insert_records(test_data, test_mappings, 'labeled_test')
                )
            except Exception as storage_error:
                logger.error(f"MongoDB storage failed: {str(storage_error)}")
                raise

            # 6. Création des index
            logger.info("Creating indexes in parallel...")
            try:
                index_operations = [
                    async_db['labeled_train'].create_index([("productid", 1)]),
                    async_db['labeled_train'].create_index([("imageid", 1)]),
                    async_db['labeled_train'].create_index([("gridfs_file_id", 1)]),
                    async_db['labeled_val'].create_index([("productid", 1)]),
                    async_db['labeled_val'].create_index([("imageid", 1)]),
                    async_db['labeled_val'].create_index([("gridfs_file_id", 1)]),
                    async_db['labeled_test'].create_index([("productid", 1)]),
                    async_db['labeled_test'].create_index([("imageid", 1)]),
                    async_db['labeled_test'].create_index([("gridfs_file_id", 1)])
                ]
                await asyncio.gather(*index_operations)
            except Exception as index_error:
                logger.error(f"Index creation failed: {str(index_error)}")
                raise

            # Vérifier les images dans GridFS
            train_count, val_count, test_count = await check_image_metadata()

            # 7. Mise à jour des métadonnées
            metadata.update({
                "status": "completed",
                "processed_records": {
                    "total": total_records,
                    "train": train_count,
                    "val": val_count,
                    "test": test_count
                },
                "image_statistics": {
                    "train": train_count,
                    "val": val_count,
                    "test": test_count
                },
                "data_distribution": {
                    "train": {str(k): v for k, v in train_data.label.value_counts().to_dict().items()},
                    "val": {str(k): v for k, v in val_data.label.value_counts().to_dict().items()},
                    "test": {str(k): v for k, v in test_data.label.value_counts().to_dict().items()}
                },
                "storage_info": {
                    "collections": {
                        "train": "labeled_train",
                        "val": "labeled_val",
                        "test": "labeled_test"
                    },
                    "gridfs": {
                        "splits": ["train", "validation", "test"],
                        "image_format": "image/jpeg",
                        "metadata_fields": ["split", "label", "productid", "imageid"]
                    }
                }
            })

            await async_db['pipeline_metadata'].insert_one(metadata)

            logger.info("Data preparation pipeline completed successfully")

            # Nettoyer les ObjectId avant de retourner la réponse
            metadata = clean_metadata(metadata)
            return {
                "message": "Data preparation completed successfully",
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error during pipeline execution: {str(e)}"
            )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Pipeline error: {error_message}", exc_info=True)
        metadata.update({
            "status": "failed",
            "error_message": error_message
        })
        
        # Nettoyer les metadata avant de les enregistrer
        clean_metadata_for_db = clean_metadata(metadata)
        await async_db['pipeline_metadata'].insert_one(clean_metadata_for_db)
        
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {error_message}"
        )

@router.post("/train-model/")
async def train_model():
    """
    Route pour l'entraînement du modèle utilisant les données labellisées de MongoDB.
    """
    try:
        logger.info("Starting model training...")
        from src.scripts.main import train_and_save_model
        
        try:
            train_and_save_model()
            logger.info("Model training completed successfully")
            return {"message": "Model training completed successfully"}
            
        except Exception as training_error:
            logger.error("Training error details:", exc_info=True)
            error_details = {
                "error_type": type(training_error).__name__,
                "error_message": str(training_error),
                "error_location": "training process"
            }
            raise HTTPException(
                status_code=500, 
                detail=f"Training process failed: {error_details}"
            )
            
    except Exception as e:
        logger.error("API endpoint error:", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during model training: {str(e)}"
        )

@router.post("/predict/")
async def predict(version: int = Query(1, description="Version number of the model to use")):
    try:
        # Utiliser MongoDB pour charger les données de test non étiquetées
        logger.info("Loading test data from 'preprocessed_x_test' collection...")
        df = pd.DataFrame(list(sync_db.preprocessed_x_test.find())).head(10)  # Limiter à 10 pour l'exemple
        logger.info(f"Loaded {len(df)} test samples")

        # Vérifier que 'productid' et 'imageid' sont présents pour récupérer les images
        required_columns = ['productid', 'imageid', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns in test data: {missing_columns}"
            )

        # Charger le prédicteur
        predictor = load_predictor(version)

        # Appel de la méthode de prédiction
        try:
            predictions = predictor.predict(df)

            if isinstance(predictions, dict) and "error" in predictions:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {predictions['error']}"
                )

            # Sauvegarder les prédictions dans MongoDB
            await async_db["predictions"].insert_one({
                "version": version,
                "date": datetime.now().isoformat(),
                "predictions": predictions
            })

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
            # 1. Chargement des données de test depuis MongoDB
            logger.info("Loading test data from 'labeled_test' collection...")
            test_data = pd.DataFrame(list(sync_db.labeled_test.find()))
            
            if test_data.empty:
                raise HTTPException(
                    status_code=404, 
                    detail="No test data found in 'labeled_test' collection"
                )
            
            # Extraire les features et labels
            X_eval = test_data.drop(['label'], axis=1)
            y_eval = test_data['label']
            
            logger.info(f"Loaded {len(X_eval)} test samples")
            
            # Optionnel : Échantillonner les données pour limiter la taille
            sample_size = min(len(X_eval), 1000)  # Limiter à 1000 échantillons par exemple
            X_eval_sample = X_eval.sample(n=sample_size, random_state=42).reset_index(drop=True)
            y_eval_sample = y_eval.loc[X_eval_sample.index].reset_index(drop=True)
            
            logger.info(f"Prepared evaluation dataset with {len(X_eval_sample)} samples")
            
        except Exception as data_error:
            logger.error(f"Data loading error: {data_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error loading test data: {str(data_error)}"
            )

        try:
            # 2. Prédictions avec le modèle chargé
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
                batch_predictions = predictor.predict(batch)
                
                # Mettre à jour les prédictions avec les indices corrects
                predictions.update({str(i): pred for i, pred in zip(range(start_idx, end_idx), batch_predictions.values())})
                
                # Log de progression
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                
                # Nettoyage après chaque lot
                del batch_predictions
                gc.collect()
            
            inference_time = (time.time() - start_time) * 1000  # Temps en millisecondes
            mean_inference_time = inference_time / len(X_eval_sample)
            
            logger.info(f"Predictions completed in {inference_time:.2f}ms")
            
        except Exception as pred_error:
            logger.error(f"Prediction error: {pred_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(pred_error)}"
            )

        try:
            # 3. Calcul des métriques
            logger.info("Computing evaluation metrics...")

            # Convertir les prédictions en un tableau numpy
            y_pred = np.array([int(predictions[str(i)]) for i in range(len(X_eval_sample))])
            y_true = y_eval_sample.values

            # Vérifier la cohérence des données
            if len(y_pred) != len(y_true):
                raise ValueError(f"Prediction length mismatch: {len(y_pred)} vs {len(y_true)}")

            # Calcul des métriques globales
            metrics = {
                "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            }

            # Initialiser les métriques par classe avec 0
            class_metrics = {f"class_{i}_f1": 0.0 for i in range(NUM_CLASSES)}

            # Calculer les métriques pour les classes présentes dans y_true
            unique_labels = np.unique(y_true)
            f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for label, score in zip(unique_labels, f1_scores):
                class_metrics[f"class_{label}_f1"] = float(score)

            # Ajouter les métriques par classe aux métriques globales
            metrics.update(class_metrics)

            logger.info("Metrics computed successfully")

        except Exception as metric_error:
            logger.error(f"Metrics calculation error: {metric_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating metrics: {str(metric_error)}"
            )


        try:
            # 4. Sauvegarde des résultats dans MongoDB
            logger.info("Saving evaluation results to MongoDB...")
            collection = sync_db["model_evaluation"]

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
            logger.info("Evaluation results saved successfully")
            
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
            detail=f"An unexpected error occurred during evaluation: {str(e)}"
        )
