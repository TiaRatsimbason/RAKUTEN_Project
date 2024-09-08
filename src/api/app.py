from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Response
from src.api.users import router as users_router, get_current_active_user, get_current_admin_user
import pandas as pd
from src.predict import Predict, load_predictor
import shutil
import subprocess
import os
import json
from tqdm import tqdm

app = FastAPI()

# Charger le prédicteur au démarrage de l'application
predictor = load_predictor()

# Inclure l'API des utilisateurs
app.include_router(users_router, prefix="/users", tags=["users"])

# Fonction pour créer les dossiers requis
def create_directory_structure(base_path="data"):
    raw_path = os.path.join(base_path, "raw")
    preprocessed_path = os.path.join(base_path, "preprocessed")

    # Créer les dossiers si nécessaire
    os.makedirs(os.path.join(raw_path, "image_train"), exist_ok=True)
    os.makedirs(os.path.join(raw_path, "image_test"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_path, "image_train"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_path, "image_test"), exist_ok=True)

# Fonction pour copier les fichiers et dossiers avec des barres de progression en temps réel
def copy_files_and_folders_from_drive(drive_path):
    try:
        data_path = os.path.join(drive_path, "molps_rakuten_data")
        for folder in ['image_train', 'image_test']:
            source = os.path.join(data_path, folder)
            dest = f"data/raw/{folder}"
            
            # Obtenir la liste des fichiers à copier
            files_to_copy = [f for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            # Créer une barre de progression
            with tqdm(total=len(files_to_copy), desc=f"Copying {folder}", unit="file") as pbar:
                for filename in files_to_copy:
                    shutil.copy(os.path.join(source, filename), dest)
                    pbar.update(1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in copying image data: {e}")

@app.post("/setup-data/")
async def setup_data():
    try:
        # Détecter le lecteur Google Drive
        drives = [f"{d}:" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")]
        google_drive_path = None
        for drive in drives:
            if os.path.exists(os.path.join(drive, "Mon Drive")):
                google_drive_path = os.path.join(drive, "Mon Drive")
                break
        
        if google_drive_path is None:
            raise HTTPException(status_code=500, detail="Google Drive not found on the system.")

        # Créer les répertoires de données
        create_directory_structure()

        # Copier les fichiers depuis Google Drive
        copy_files_and_folders_from_drive(google_drive_path)

        # Exécuter le script pour importer les données
        subprocess.run(["python", "src/data/import_raw_data.py"], check=True)

        # Exécuter le script pour créer le dataset
        subprocess.run(["python", "src/data/make_dataset.py", "data/raw", "data/preprocessed"], check=True)

        return {"message": "Data setup completed successfully."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in setting up data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in downloading or copying data: {e}")

@app.post("/train-model/")
async def train_model(current_user: dict = Depends(get_current_admin_user)):
    try:
        # Exécuter le script main.py pour entraîner le modèle
        subprocess.run(["python", "src/main.py"], check=True)
        return {"message": "Model training completed successfully."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in training model: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during model training: {e}")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    images_folder: str = "data/preprocessed/image_test",
    current_user: dict = Depends(get_current_active_user),  # Protéger cette route avec l'authentification
):
    # Sauvegarder le fichier temporairement
    with open("temp.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Lire le fichier CSV et le convertir en DataFrame
    df = pd.read_csv("temp.csv")[:10]
    
    # Appel de la méthode de prédiction
    predictions = predictor.predict(df, images_folder)
    
    # Sauvegarder les prédictions dans un fichier JSON dans le répertoire "data/preprocessed"
    output_path = "data/preprocessed/predictions.json"
    with open(output_path, "w") as json_file:
        json.dump(predictions, json_file, indent=2)
    
    # Supprimer le fichier temporaire après utilisation
    os.remove("temp.csv")
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





