import os
import shutil
from tqdm import tqdm


def create_directory_structure(base_path="data"):
    raw_path = os.path.join(base_path, "raw")
    preprocessed_path = os.path.join(base_path, "preprocessed")

    # Créer les dossiers si nécessaire
    os.makedirs(os.path.join(raw_path, "image_train"), exist_ok=True)
    os.makedirs(os.path.join(raw_path, "image_test"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_path, "image_train"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_path, "image_test"), exist_ok=True)


def copy_files_and_folders_from_drive(drive_path):
    try:
        data_path = os.path.join(drive_path, "molps_rakuten_data")
        for folder in ["image_train", "image_test"]:
            source = os.path.join(data_path, folder)
            dest = f"data/raw/{folder}"

            # Obtenir la liste des fichiers à copier
            files_to_copy = [
                f
                for f in os.listdir(source)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
            ]

            # Créer une barre de progression
            with tqdm(
                total=len(files_to_copy), desc=f"Copying {folder}", unit="file"
            ) as pbar:
                for filename in files_to_copy:
                    shutil.copy(os.path.join(source, filename), dest)
                    pbar.update(1)

    except Exception as e:
        print(f"Error in copying image data: {e}")
        raise e


def main():
    try:
        # Détecter le lecteur Google Drive
        drives = [
            f"{d}:" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")
        ]
        google_drive_path = None
        for drive in drives:
            if os.path.exists(os.path.join(drive, "Mon Drive")):
                google_drive_path = os.path.join(drive, "Mon Drive")
                break

        if google_drive_path is None:
            print("Google Drive not found on the system.")
            return

        # Créer les répertoires de données
        create_directory_structure()

        # Copier les fichiers depuis Google Drive
        copy_files_and_folders_from_drive(google_drive_path)

        # Exécuter le script pour importer les données
        os.system("python src/scripts/data/import_raw_data.py")

        # Exécuter le script pour créer le dataset
        os.system("python src/scripts/data/make_dataset.py data/raw data/preprocessed")

        print("Data setup completed successfully.")

    except Exception as e:
        print(f"Error in setting up data: {e}")


if __name__ == "__main__":
    main()
