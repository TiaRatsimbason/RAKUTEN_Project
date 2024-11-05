import pandas as pd
import gridfs
import os
from pymongo import MongoClient
from tqdm import tqdm
import logging
from typing import List, Dict
import numpy as np
from PIL import Image
import io

class MongoDBDataLoader:
    def __init__(self, mongodb_uri: str = "mongodb://admin:motdepasseadmin@mongo:27017/"):
        """
        Initialise la connexion à MongoDB et configure le logging
        """
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['rakuten_db']
        self.fs = gridfs.GridFS(self.db)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_csv_to_mongodb(self, csv_path: str, collection_name: str) -> bool:
        """
        Charge un fichier CSV dans une collection MongoDB
        """
        try:
            # Lecture du CSV
            df = pd.read_csv(csv_path)
            
            # Conversion des données pour MongoDB
            records = df.replace({np.nan: None}).to_dict('records')
            
            # Suppression des données existantes
            self.db[collection_name].drop()
            
            # Insertion des nouvelles données
            self.db[collection_name].insert_many(records)
            
            self.logger.info(f"CSV {csv_path} chargé avec succès dans {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {csv_path}: {str(e)}")
            return False

    def load_image_to_gridfs(self, image_path: str, productid: str, imageid: str) -> str:
        """
        Charge une image dans GridFS
        """
        try:
            # Ouverture et compression de l'image
            with Image.open(image_path) as img:
                # Conversion en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Sauvegarde en buffer avec compression
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)
                
                # Métadonnées de l'image
                metadata = {
                    'productid': productid,
                    'imageid': imageid,
                    'original_path': image_path
                }
                
                # Stockage dans GridFS
                file_id = self.fs.put(
                    buffer.getvalue(),
                    filename=f"image_{imageid}_product_{productid}.jpg",
                    metadata=metadata
                )
                
                return str(file_id)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de l'image {image_path}: {str(e)}")
            return None

    def load_image_folder_to_gridfs(self, folder_path: str) -> Dict[str, List[str]]:
        """
        Charge un dossier d'images dans GridFS
        """
        results = {'success': [], 'failed': []}
        
        try:
            # Parcours des images dans le dossier
            for filename in tqdm(os.listdir(folder_path), desc=f"Chargement des images de {folder_path}"):
                if filename.endswith('.jpg'):
                    # Extraction des IDs depuis le nom du fichier
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 4:
                        imageid = parts[1]
                        productid = parts[3]
                        
                        image_path = os.path.join(folder_path, filename)
                        file_id = self.load_image_to_gridfs(image_path, productid, imageid)
                        
                        if file_id:
                            results['success'].append(filename)
                        else:
                            results['failed'].append(filename)
            
            self.logger.info(f"Chargement terminé pour {folder_path}: "
                           f"{len(results['success'])} succès, {len(results['failed'])} échecs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du dossier {folder_path}: {str(e)}")
            return results

    def load_all_data(self, preprocessed_path: str = "/app/data/preprocessed"):
        """
        Charge toutes les données preprocessed dans MongoDB
        """
        try:
            # Configuration des chemins
            csv_files = {
                'X_test': os.path.join(preprocessed_path, 'X_test_update.csv'),
                'X_train': os.path.join(preprocessed_path, 'X_train_update.csv'),
                'Y_train': os.path.join(preprocessed_path, 'Y_train_CVw08PX.csv')
            }
            
            image_folders = {
                'test_images': os.path.join(preprocessed_path, 'image_test'),
                'train_images': os.path.join(preprocessed_path, 'image_train')
            }
            
            # Chargement des CSVs
            for name, path in csv_files.items():
                self.load_csv_to_mongodb(path, f"preprocessed_{name.lower()}")
            
            # Chargement des images
            for name, path in image_folders.items():
                self.load_image_folder_to_gridfs(path)
            
            # Création des index
            self.db.preprocessed_x_train.create_index("productid")
            self.db.preprocessed_x_test.create_index("productid")
            self.db.preprocessed_y_train.create_index("productid")
            
            self.logger.info("Chargement complet des données terminé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement complet des données: {str(e)}")
            raise

if __name__ == "__main__":
    loader = MongoDBDataLoader()
    loader.load_all_data()