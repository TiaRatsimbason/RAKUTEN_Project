import os
import shutil
import logging
from check_structure import check_existing_file, check_existing_folder
import time  # Pour le suivi du temps d'exécution

def import_raw_data(raw_data_relative_path, filenames, local_folder_path):
    """Import filenames from local_folder_path in raw_data_relative_path"""
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path, exist_ok=True)

    # Limiter à 100 fichiers pour accélérer l'exécution
    filenames = filenames[:100]

    # Copier les fichiers un par un
    start_time = time.time()  # Démarrer le chronomètre
    for filename in filenames:
        input_file = os.path.join(local_folder_path, filename)
        output_file = os.path.join(raw_data_relative_path, filename)

        if not os.path.exists(output_file):
            print(f"Copying {input_file} as {os.path.basename(output_file)}")
            shutil.copy(input_file, output_file)
        else:
            print(f"File {output_file} already exists, skipping.")

    logging.info(f"Copie des fichiers terminée en {time.time() - start_time} secondes")

    # Copier le dossier 'image_train'
    img_train_folder = os.path.join(local_folder_path, "image_train/")
    img_train_local_path = os.path.join(raw_data_relative_path, "image_train/")
    
    if check_existing_folder(img_train_local_path):
        os.makedirs(img_train_local_path, exist_ok=True)

    start_time = time.time()  # Recommencer le chronomètre pour la copie des images
    if os.path.exists(img_train_folder):
        img_filenames = os.listdir(img_train_folder)[:100]  # Limiter à 100 images

        for img_filename in img_filenames:
            input_image = os.path.join(img_train_folder, img_filename)
            output_image = os.path.join(img_train_local_path, img_filename)
            if not os.path.exists(output_image):
                print(f"Copying {input_image} to {output_image}")
                shutil.copy(input_image, output_image)
            else:
                print(f"Image {output_image} already exists, skipping.")
    else:
        print(f"Folder {img_train_folder} does not exist")

    logging.info(f"Copie des images terminée en {time.time() - start_time} secondes")


def main(
    raw_data_relative_path="C:/Users/Elsa/Documents/datascientest/projet_rakuten/juin24cmlops_rakuten_2/data/raw",
    filenames=["X_test_update.csv", "X_train_update.csv", "Y_train_CVw08PX.csv"],
    local_folder_path="C:/Users/Elsa/Documents/datascientest/projet_rakuten/juin24cmlops_rakuten_2/data/preprocessed"
):
    """Copy data from local paths"""
    import_raw_data(raw_data_relative_path, filenames, local_folder_path)
    logger = logging.getLogger(__name__)
    logger.info("making raw data set")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
