# Standard library imports
from datetime import datetime  # Changement ici
import json
import os
import logging
import os

# Third-party imports
import pandas as pd
import numpy as np
import keras
import mlflow
from sklearn.metrics import accuracy_score
from bson import ObjectId

# Local imports
from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.scripts.models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
from src.config.mongodb import sync_db, async_db, sync_fs, async_fs

# Suppression du warning Git MLflow
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# Constantes
NUM_CLASSES = 27
BATCH_SIZE = 32

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    """
    Fonction principale d'entraînement qui utilise les données labellisées de MongoDB.
    """
    # Configurer MLflow
    mlflow.set_tracking_uri("http://mlflow-ui:5000")
    experiment_name = "Rakuten Model Training"
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"Train Models {timestamp}") as main_run:
        try:
            # 1. Vérifier et charger les données labellisées
            collections_required = ['labeled_train', 'labeled_val']
            if not all(col in sync_db.list_collection_names() for col in collections_required):
                raise ValueError("Required labeled collections not found in MongoDB")

            # Vérifier que les images sont disponibles
            train_images = sync_db.fs.files.count_documents({"metadata.split": "train"})
            val_images = sync_db.fs.files.count_documents({"metadata.split": "validation"})
            
            if train_images == 0 or val_images == 0:
                raise ValueError("No images found in GridFS for training/validation splits")
            
            logger.info(f"Found {train_images} training and {val_images} validation images in GridFS")

            # Charger les données avec les références GridFS
            train_data = pd.DataFrame(list(sync_db.labeled_train.find({}, {'_id': 0})))
            val_data = pd.DataFrame(list(sync_db.labeled_val.find({}, {'_id': 0})))
            
            # Séparer features et labels
            X_train = train_data.drop(['label', 'gridfs_file_id'], axis=1)
            y_train = train_data['label']
            X_val = val_data.drop(['label', 'gridfs_file_id'], axis=1)
            y_val = val_data['label']
            
            # Ajouter les références GridFS pour les images
            X_train['gridfs_image_ref'] = train_data['gridfs_file_id']
            X_val['gridfs_image_ref'] = val_data['gridfs_file_id']
            
            logger.info(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")

            # 2. Prétraiter les données texte
            text_preprocessor = TextPreprocessor()
            text_preprocessor.preprocess_text_in_df(X_train, ["description"])
            text_preprocessor.preprocess_text_in_df(X_val, ["description"])

            # 3. Prétraiter les images
            image_preprocessor = ImagePreprocessor(image_type="train")
            image_preprocessor.preprocess_images_in_df(X_train)
            image_preprocessor.preprocess_images_in_df(X_val)

            # 4. Enregistrer les paramètres d'entraînement
            mlflow.log_param("num_classes", NUM_CLASSES)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("max_sequence_length", 10)

            # 5. Entraîner et sauvegarder les modèles
            from src.scripts.models.train_model import train_and_save_models
            train_and_save_models(X_train, y_train, X_val, y_val)

            logger.info("Model training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise