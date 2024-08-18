from fastapi import FastAPI, UploadFile, File
import pandas as pd
import json
from typing import List
from pathlib import Path

app = FastAPI()

# Charger le modèle et les autres composants nécessaires
# Exemple : 
# model = load_model('models/trained_model.pkl')

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de recommandation de films !"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lire le fichier envoyé et le convertir en DataFrame
    df = pd.read_csv(file.file)

    # Appeler la fonction de prédiction (par exemple celle dans predict.py)
    # predictions = predict(df) # Remplacer par la vraie fonction de prédiction

    # Simuler une réponse de prédiction pour l'exemple
    predictions = {"prediction": [1, 2, 3, 4, 5]}  # À remplacer par les vraies prédictions

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
