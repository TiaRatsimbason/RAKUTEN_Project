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
import os

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataImporter:
    def __init__(self, filepath="/app/data/preprocessed"):
        self.filepath = filepath
        logger.info(f"DataImporter initialized with filepath: {self.filepath}")

    def load_data(self):
        logger.info("Loading data files...")
        data = pd.read_csv(f"{self.filepath}/X_train_update.csv")
        data["description"] = data["designation"] + " " + data["description"].fillna("")
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(f"{self.filepath}/Y_train_CVw08PX.csv")
        target = target.drop(["Unnamed: 0"], axis=1)

        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
        }
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        os.makedirs("models", exist_ok=True)
        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)

        df = pd.concat([data, target], axis=1)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def split_train_test(self, df):
        logger.info("Distribution des classes:\n%s", df['prdtypecode'].value_counts())
        
        grouped_data = df.groupby("prdtypecode")
        X_train_samples = []
        X_val_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            n_samples = len(group)
            if n_samples < 3:
                X_train_samples.append(group)  # Tout mettre dans train si peu d'échantillons
                continue
                
            n_train = max(int(n_samples * 0.7), 1)
            n_val = max(int(n_samples * 0.15), 1)
            
            shuffled = group.sample(frac=1, random_state=42)
            X_train_samples.append(shuffled[:n_train])
            X_val_samples.append(shuffled[n_train:n_train + n_val])
            X_test_samples.append(shuffled[n_train + n_val:])

        X_train = pd.concat(X_train_samples).sample(frac=1, random_state=42).reset_index(drop=True)
        X_val = pd.concat(X_val_samples).sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = pd.concat(X_test_samples).sample(frac=1, random_state=42).reset_index(drop=True)

        y_train = X_train["prdtypecode"]
        X_train = X_train.drop(["prdtypecode"], axis=1)

        y_val = X_val["prdtypecode"]
        X_val = X_val.drop(["prdtypecode"], axis=1)

        y_test = X_test["prdtypecode"]
        X_test = X_test.drop(["prdtypecode"], axis=1)

        logger.info(f"Taille train set: {len(X_train)}")
        logger.info(f"Taille validation set: {len(X_val)}")
        logger.info(f"Taille test set: {len(X_test)}")
        logger.info(f"Distribution des classes dans train:\n{y_train.value_counts()}")

        return X_train, X_val, X_test, y_train, y_val, y_test

class ImagePreprocessor:
    def __init__(self, filepath="/app/data/preprocessed/image_train"):
        self.filepath = filepath
        logger.info(f"ImagePreprocessor initialized with filepath: {self.filepath}")

    def preprocess_images_in_df(self, df):
        image_paths = []
        for _, row in df.iterrows():
            image_path = os.path.join(
                self.filepath,
                f"image_{row['imageid']}_product_{row['productid']}.jpg"
            )
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
            image_paths.append(image_path)
        
        df["image_path"] = image_paths
        logger.info(f"Processed {len(df)} image paths")
        logger.info(f"Sample image path: {image_paths[0] if image_paths else 'No images found'}")

class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))
        logger.info("TextPreprocessor initialized")

    def preprocess_text(self, text):
        if isinstance(text, float) and math.isnan(text):
            return ""

        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = word_tokenize(text.lower())
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]
        return " ".join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)
        logger.info(f"Preprocessed text columns: {columns}")