import os
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
import logging
import json
from src.config.mongodb import sync_db, sync_fs

logger = logging.getLogger(__name__)

class DataImporter:
    def __init__(self):
        """
        Initialise DataImporter pour MongoDB uniquement
        """
        self.db = sync_db
        self.fs = sync_fs

    def load_data(self):
        try:
            logger.info("Loading data from MongoDB...")
            # Charger les données d'entraînement
            x_train = pd.DataFrame(list(self.db.preprocessed_x_train.find({}, {'_id': 0})))
            y_train = pd.DataFrame(list(self.db.preprocessed_y_train.find({}, {'_id': 0})))
            
            # Prétraitement du texte
            x_train["description"] = x_train["designation"] + x_train["description"].fillna('')
            x_train = x_train.drop(["designation"], axis=1)
            
            # Mapper les modalités exactement comme avant mais en convertissant en int standard
            modalite_mapping = {
                int(modalite): int(i) for i, modalite in enumerate(y_train["prdtypecode"].unique())
            }
            
            # Convertir les clés et valeurs en str pour le JSON
            modalite_mapping_str = {str(k): str(v) for k, v in modalite_mapping.items()}
            
            # Appliquer le mapping (en gardant les valeurs numériques pour l'entraînement)
            y_train["prdtypecode"] = y_train["prdtypecode"].map(modalite_mapping)
            
            # Sauvegarder le mapping (version str)
            os.makedirs("models", exist_ok=True)
            with open("models/mapper.json", "w") as fichier:
                json.dump(modalite_mapping_str, fichier, indent=2)
            logger.info("Saved category mapping to models/mapper.json")
            
            return pd.concat([x_train, y_train], axis=1)
            
        except Exception as e:
            logger.error(f"Failed to load data from MongoDB: {e}")
            raise

    def split_train_test(self, df, samples_per_class=600):
        """
        Split les données en ensembles d'entraînement, validation et test
        """
        logger.info(f"Splitting data with {samples_per_class} samples per class...")
        
        grouped_data = df.groupby("prdtypecode")
        X_train_samples = []
        X_test_samples = []

        for label, group in grouped_data:
            if len(group) < samples_per_class:
                logger.warning(f"Class {label} has less than {samples_per_class} samples")
                samples = group.sample(n=min(len(group), samples_per_class), random_state=42)
            else:
                samples = group.sample(n=samples_per_class, random_state=42)
            
            X_train_samples.append(samples)
            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples)

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)

        # Mélanger les données
        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=42).reset_index(drop=True)

        # Séparer les features et les labels
        y_train = X_train["prdtypecode"]
        X_train = X_train.drop(["prdtypecode"], axis=1)

        y_test = X_test["prdtypecode"]
        X_test = X_test.drop(["prdtypecode"], axis=1)

        # Créer l'ensemble de validation
        val_samples_per_class = 50
        grouped_data_test = pd.concat([X_test, y_test], axis=1).groupby("prdtypecode")

        X_val_samples = []
        y_val_samples = []

        for _, group in grouped_data_test:
            samples = group.sample(n=val_samples_per_class, random_state=42)
            X_val_samples.append(samples[["description", "productid", "imageid"]])
            y_val_samples.append(samples["prdtypecode"])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        # Mélanger l'ensemble de validation
        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Split completed: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, image_type="train"):
        """
        Initialise ImagePreprocessor pour utiliser GridFS
        
        Args:
            image_type (str): "train" ou "test" pour spécifier quel ensemble d'images utiliser
        """
        self.fs = sync_fs
        self.image_type = image_type
        logger.info(f"Initialized ImagePreprocessor for {image_type} images")
        
    def preprocess_images_in_df(self, df):
        """
        Prétraite les images en utilisant GridFS selon le type (train/test)
        """
        logger.info(f"Loading and preprocessing {self.image_type} images from GridFS...")
        from src.scripts.data.gridfs_image_handler import GridFSImageHandler
        
        image_pattern = f"/image_{self.image_type}/"
        
        # Vérifier que les images existent dans GridFS
        count = sync_db.fs.files.count_documents({"metadata.original_path": {"$regex": image_pattern}})
        logger.info(f"Found {count} {self.image_type} images in GridFS")
        
        with GridFSImageHandler() as handler:
            # Extraire les images du bon type
            image_paths = handler.batch_extract_images(df, image_type=self.image_type)
            df["image_path"] = df.apply(
                lambda row: image_paths.get(f"{row['imageid']}_{row['productid']}", None),
                axis=1
            )
            
        # Vérifier que toutes les images ont été trouvées
        missing_images = df[df["image_path"].isna()]
        if not missing_images.empty:
            logger.warning(f"Missing {len(missing_images)} {self.image_type} images in GridFS")
            logger.warning(f"First few missing images: {missing_images[['imageid', 'productid']].head()}")
            
        logger.info(f"{self.image_type} image preprocessing completed")


class TextPreprocessor:
    def __init__(self):
        logger.info("Initializing text preprocessor...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('french'))
        
    def preprocess_text(self, text):
        """
        Prétraite un texte
        """
        if pd.isna(text) or text is None:
            logger.warning("Found null text value, replacing with empty string")
            return ""
            
        try:
            # Conversion en string si nécessaire
            text = str(text)
            
            # Nettoyage HTML
            text = BeautifulSoup(text, "html.parser").get_text()
            
            # Conversion en minuscules
            text = text.lower()
            
            # Suppression de la ponctuation et des chiffres
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Tokenization
            words = word_tokenize(text)
            
            # Suppression des stop words
            words = [word for word in words if word not in self.stop_words]
            
            # Rejoindre les mots
            text = ' '.join(words)
            
            # Supprimer les espaces multiples
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return ""

    def preprocess_text_in_df(self, df, columns):
        """
        Prétraite les colonnes textuelles d'un DataFrame
        """
        logger.info(f"Preprocessing text columns: {columns}")
        try:
            # Vérifier les valeurs nulles
            for column in columns:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} null values in column {column}")
                
                # Remplacer les valeurs nulles par une chaîne vide
                df[column] = df[column].fillna("")
                
                # Appliquer le prétraitement
                df[column] = df[column].apply(self.preprocess_text)
                
            logger.info("Text preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error in preprocess_text_in_df: {str(e)}", exc_info=True)
            raise

    def __call__(self, text):
        """
        Permet d'utiliser la classe comme une fonction
        """
        return self.preprocess_text(text)