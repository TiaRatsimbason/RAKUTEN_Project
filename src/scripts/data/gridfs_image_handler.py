import os
import tempfile
import logging
from typing import Dict
import gridfs
from src.config.mongodb import sync_db, async_db, sync_fs, async_fs

# Configuration du logger
logger = logging.getLogger(__name__)

class GridFSImageHandler:
    def __init__(self):
        self.db = sync_db
        self.fs = sync_fs
        self.async_fs = async_fs
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

    def get_image_path(self, imageid, productid, image_type="train"):
        """
        Récupère l'image depuis GridFS selon son type
        """
        try:
            # Chercher l'image avec le bon type (train/test)
            image_pattern = f"/image_{image_type}/"
            image_file = self.fs.find_one({
                "metadata.imageid": str(imageid),
                "metadata.productid": str(productid),
                "metadata.original_path": {"$regex": image_pattern}
            })
            
            if not image_file:
                raise FileNotFoundError(
                    f"{image_type} image not found for imageid={imageid}, productid={productid}"
                )
            return image_file._id
            
        except Exception as e:
            logger.error(f"Error retrieving {image_type} image from GridFS: {e}")
            raise

    def batch_extract_images(self, df, image_type="train") -> Dict[str, str]:
        """
        Extrait un lot d'images selon leur type et les sauvegarde dans un dossier temporaire
        """
        logger.info(f"Batch extracting {image_type} images from GridFS")
        image_paths = {}
        
        for _, row in df.iterrows():
            try:
                # Obtenir l'ID de l'image
                file_id = self.get_image_path(
                    row['imageid'], 
                    row['productid'],
                    image_type=image_type
                )
                
                # Créer le chemin temporaire pour l'image
                temp_path = os.path.join(self.temp_dir, f"{row['imageid']}_{row['productid']}.jpg")
                
                # Récupérer et sauvegarder l'image
                grid_out = self.fs.get(file_id)
                with open(temp_path, 'wb') as f:
                    f.write(grid_out.read())
                
                image_paths[f"{row['imageid']}_{row['productid']}"] = temp_path
                
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
                continue
                
        return image_paths