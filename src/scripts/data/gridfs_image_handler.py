import os
import tempfile
import logging
from typing import Dict
from bson import ObjectId
from PIL import Image
from gridfs import GridFS
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
            df (pd.DataFrame): DataFrame contenant la colonne 'gridfs_file_id'

        Returns:
            Dict[str, str]: Dictionnaire mapping gridfs_file_id à chemin local de l'image extraite
        """
        logger.info(f"Batch extracting images from GridFS using gridfs_file_id")
        image_paths = {}

        for idx, row in df.iterrows():
            gridfs_file_id = row.get('gridfs_file_id')
            if not gridfs_file_id:
                logger.warning(f"No gridfs_file_id found for row with index {idx}")
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

        return image_paths

    def batch_extract_images_by_ids(self, df) -> Dict[str, str]:
        """
        Extrait un lot d'images en utilisant 'productid' et 'imageid' et les sauvegarde dans un dossier temporaire

        Args:
            df (pd.DataFrame): DataFrame contenant les colonnes 'productid' et 'imageid'

        Returns:
            Dict[str, str]: Dictionnaire mapping 'productid_imageid' à chemin local de l'image extraite
        """
        logger.info(f"Batch extracting images from GridFS using productid and imageid")
        image_paths = {}

        for idx, row in df.iterrows():
            product_id = str(row['productid'])
            image_id = str(row['imageid'])
            key = f"{product_id}_{image_id}"

            try:
                # Rechercher l'image dans GridFS
                file_doc = self.fs.find_one({
                    "metadata.productid": product_id,
                    "metadata.imageid": image_id
                })

                if file_doc:
                    file_id = file_doc._id
                    # Créer le chemin temporaire pour l'image
                    temp_path = os.path.join(self.temp_dir, f"{key}.jpg")

                    # Sauvegarder l'image
                    with open(temp_path, 'wb') as f:
                        f.write(file_doc.read())

                    image_paths[key] = temp_path
                    logger.debug(f"Successfully extracted image for key {key}")
                else:
                    logger.warning(f"No image found in GridFS for key {key}")

            except Exception as e:
                logger.warning(f"Failed to extract image with key {key}: {e}")
                continue

        return image_paths

    def extract_image_to_temp(self, file_id) -> str:
        """
        Extrait une image spécifique depuis GridFS et la sauvegarde dans un fichier temporaire.

        Args:
            file_id (ObjectId): L'identifiant du fichier dans GridFS.

        Returns:
            str: Le chemin du fichier temporaire où l'image a été sauvegardée.
        """
        try:
            grid_out = self.fs.get(file_id)
            if not grid_out:
                raise FileNotFoundError(f"Image not found in GridFS for file_id={file_id}")

            # Créer le chemin temporaire pour l'image
            temp_path = os.path.join(self.temp_dir, f"{file_id}.jpg")

            # Sauvegarder l'image
            with open(temp_path, 'wb') as f:
                f.write(grid_out.read())

            return temp_path

        except Exception as e:
            logger.warning(f"Failed to extract image with file_id {file_id}: {e}")
            return None
