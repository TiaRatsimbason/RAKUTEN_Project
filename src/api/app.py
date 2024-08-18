from fastapi import FastAPI, UploadFile, File
import pandas as pd
from src.predict import Predict, load_predictor
import shutil

app = FastAPI()

# Charger le prédicteur au démarrage de l'application
predictor = load_predictor()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), images_folder: str = "data/preprocessed/image_test"):
    # Sauvegarder le fichier temporairement
    with open("temp.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Lire le fichier CSV et le convertir en DataFrame
    df = pd.read_csv("temp.csv")
    
    # Appel de la méthode de prédiction
    predictions = predictor.predict(df, images_folder)
    
    # Supprimer le fichier temporaire après utilisation
    os.remove("temp.csv")
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


