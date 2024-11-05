import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
from src.config.mongodb import sync_db, sync_fs
import os

class DataImporter:
    def __init__(self, filepath="data/preprocessed", use_mongodb=True, mongodb_uri="mongodb://admin:motdepasseadmin@mongo:27017/"):
        """
        Initialise DataImporter avec support pour MongoDB et fichiers locaux
        
        Args:
            filepath (str): Chemin vers les données locales
            use_mongodb (bool): Utiliser MongoDB au lieu des fichiers locaux
            mongodb_uri (str): URI de connexion MongoDB
        """
        self.filepath = filepath
        self.use_mongodb = use_mongodb
        
        if use_mongodb:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client['rakuten_db']
            self.fs = gridfs.GridFS(self.db)

    def load_data(self):
        """
        Charge les données depuis MongoDB ou les fichiers locaux selon la configuration
        """
        if self.use_mongodb:
            try:
                # Charger les données depuis MongoDB
                x_train = pd.DataFrame(list(self.db.preprocessed_x_train.find({}, {'_id': 0})))
                y_train = pd.DataFrame(list(self.db.preprocessed_y_train.find({}, {'_id': 0})))
                
                # Combiner description et designation
                x_train["description"] = x_train["designation"] + x_train["description"].fillna('')
                x_train = x_train.drop(["designation"], axis=1)
                
            except Exception as e:
                print(f"Erreur lors du chargement depuis MongoDB: {e}")
                print("Utilisation du chargement depuis les fichiers locaux comme fallback...")
                self.use_mongodb = False
                return self._load_data_from_files()
        else:
            return self._load_data_from_files()
        
        # Mapper les modalités (commun aux deux méthodes)
        target = y_train["prdtypecode"]
        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target.unique())
        }
        y_train["prdtypecode"] = y_train["prdtypecode"].replace(modalite_mapping)
        
        # Sauvegarder le mapping
        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)
            
        return pd.concat([x_train, y_train], axis=1)

    def _load_data_from_files(self):
        """
        Méthode privée pour charger les données depuis les fichiers locaux
        """
        data = pd.read_csv(f"{self.filepath}/X_train_update.csv")
        data["description"] = data["designation"] + str(data["description"])
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(f"{self.filepath}/Y_train_CVw08PX.csv")
        target = target.drop(["Unnamed: 0"], axis=1)
        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
        }
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)

        return pd.concat([data, target], axis=1)

    def get_image_path(self, imageid, productid):
        """
        Récupère le chemin de l'image depuis MongoDB ou le système de fichiers
        """
        if self.use_mongodb:
            image_file = self.fs.find_one({
                "metadata.imageid": str(imageid),
                "metadata.productid": str(productid)
            })
            if image_file:
                return image_file._id
        
        # Fallback vers le chemin local
        return os.path.join(
            self.filepath,
            "image_train",
            f"image_{imageid}_product_{productid}.jpg"
        )

    def split_train_test(self, df, samples_per_class=600):
        """
        Split les données en ensembles d'entraînement, validation et test
        La méthode reste inchangée car elle travaille sur le DataFrame déjà chargé
        """
        grouped_data = df.groupby("prdtypecode")

        X_train_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            samples = group.sample(n=samples_per_class, random_state=42)
            X_train_samples.append(samples)

            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples)

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)

        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=42).reset_index(drop=True)

        y_train = X_train["prdtypecode"]
        X_train = X_train.drop(["prdtypecode"], axis=1)

        y_test = X_test["prdtypecode"]
        X_test = X_test.drop(["prdtypecode"], axis=1)

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

        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, data_importer=None):
        self.data_importer = data_importer
        
    def preprocess_images_in_df(self, df):
        """
        Prétraite les images avec gestion de GridFS
        """
        if self.data_importer and self.data_importer.use_mongodb:
            from src.scripts.data.gridfs_image_handler import GridFSImageHandler
            with GridFSImageHandler() as handler:
                # Créer un mapping d'images temporaires
                image_paths = handler.batch_extract_images(df)
                df["image_path"] = df.apply(
                    lambda row: image_paths.get(f"{row['imageid']}_{row['productid']}", None),
                    axis=1
                )
        else:
            # Ancien code pour les fichiers locaux
            filepath = "data/preprocessed/image_train"
            df["image_path"] = (
                f"{filepath}/image_"
                + df["imageid"].astype(str)
                + "_product_"
                + df["productid"].astype(str)
                + ".jpg"
            )

class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words("french")
        )  # Vous pouvez choisir une autre langue si nécessaire

    def preprocess_text(self, text):

        if isinstance(text, float) and math.isnan(text):
            return ""
        # Supprimer les balises HTML
        text = BeautifulSoup(text, "html.parser").get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)
