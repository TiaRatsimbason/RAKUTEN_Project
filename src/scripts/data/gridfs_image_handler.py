import os
import tempfile
import logging
from typing import Dict
from bson import ObjectId
from PIL import Image
import gridfs
from src.config.mongodb import sync_db, sync_fs

# Configuration du logger
logger = logging.getLogger(__name__)

class GridFSImageHandler:
    def __init__(self):
        self.db = sync_db
        self.fs = sync_fs
        self.temp_dir = None

    def __enter__(self):
        """Méthode appelée quand on entre dans le bloc 'with'"""
        self.temp_dir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Méthode appelée quand on sort du bloc 'with'"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def get_image_by_file_id(self, gridfs_file_id):
        """
        Récupère l'image depuis GridFS en utilisant gridfs_file_id
        """
        try:
            grid_out = self.fs.get(ObjectId(gridfs_file_id))
            if not grid_out:
                raise FileNotFoundError(
                    f"Image not found in GridFS for file_id={gridfs_file_id}"
                )
            return grid_out
        except Exception as e:
            logger.error(f"Error retrieving image from GridFS with file_id {gridfs_file_id}: {e}")
            raise

    def batch_extract_images(self, df) -> Dict[str, str]:
        """
        Extrait un lot d'images en utilisant gridfs_file_id et les sauvegarde dans un dossier temporaire

        Args:
            df (pd.DataFrame): DataFrame contenant les colonnes 'gridfs_image_ref'

        Returns:
            Dict[str, str]: Dictionnaire mapping gridfs_file_id à chemin local de l'image extraite
        """
        logger.info(f"Batch extracting images from GridFS using gridfs_file_id")
        image_paths = {}
        
        for _, row in df.iterrows():
            gridfs_file_id = row.get('gridfs_image_ref')
            if not gridfs_file_id:
                logger.warning(f"No gridfs_file_id found for row with index {_}")
                continue

            try:
                # Récupérer l'image depuis GridFS
                grid_out = self.fs.get(ObjectId(gridfs_file_id))
                if not grid_out:
                    raise FileNotFoundError(f"Image not found in GridFS for file_id={gridfs_file_id}")
                
                # Créer le chemin temporaire pour l'image
                temp_path = os.path.join(self.temp_dir, f"{gridfs_file_id}.jpg")
                
                # Sauvegarder l'image
                with open(temp_path, 'wb') as f:
                    f.write(grid_out.read())
                
                image_paths[str(gridfs_file_id)] = temp_path
                
            except Exception as e:
                logger.warning(f"Failed to extract image with file_id {gridfs_file_id}: {e}")
                continue
                
        return image_paths  # Ajout de cette ligne pour renvoyer le dictionnaire des chemins d'images

    def batch_extract_images_by_ids(self, df) -> Dict[str, str]:
        """
        Extrait un lot d'images en utilisant imageid et productid et les sauvegarde dans un dossier temporaire
        """
        logger.info(f"Batch extracting images from GridFS using imageid and productid")
        image_paths = {}
        
        for _, row in df.iterrows():
            imageid = str(row['imageid'])
            productid = str(row['productid'])
            try:
                # Rechercher l'image dans GridFS
                grid_out = self.fs.find_one({
                    "metadata.imageid": imageid,
                    "metadata.productid": productid,
                    "metadata.original_path": {"$regex": "/image_"}
                })
                if not grid_out:
                    raise FileNotFoundError(f"Image not found in GridFS for imageid={imageid}, productid={productid}")
                
                # Créer le chemin temporaire pour l'image
                temp_path = os.path.join(self.temp_dir, f"{imageid}_{productid}.jpg")
                
                # Sauvegarder l'image
                with open(temp_path, 'wb') as f:
                    f.write(grid_out.read())
                
                image_paths[f"{imageid}_{productid}"] = temp_path
                
            except Exception as e:
                logger.warning(f"Failed to extract image with imageid={imageid}, productid={productid}: {e}")
                continue
            
        return image_paths
