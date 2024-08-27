from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from src.api.users import router as users_router, get_current_active_user
import pandas as pd
from src.predict import Predict, load_predictor
import shutil
import os
import json

app = FastAPI()

# Charger le prédicteur au démarrage de l'application
predictor = load_predictor()

# Inclure l'API des utilisateurs
app.include_router(users_router, prefix="/users", tags=["users"])

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
    

# Pour enregistrer un utilisateur:  
"""
$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "username" = "Tia"
    "password" = "Tia@7777"
    "role" = "user"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/users/register/" `
                  -Method POST `
                  -Headers $headers `
                  -Body $body


"""


# Pour avoir l'access_token: 

"""
$headers = @{
    "Content-Type" = "application/x-www-form-urlencoded"
}

$body = "username=Tia&password=Tia@7777"

$response = Invoke-RestMethod -Uri "http://localhost:8000/users/token" `
                              -Method POST `
                              -Headers $headers `
                              -Body $body

$token = $response.access_token

"""


# Pour faire une requête à l'api:

"""
$headers = @{
    "Authorization" = "Bearer $token"
    "accept" = "application/json"
}

$form = @{
    "file" = Get-Item "C:/Users/Tia/Documents/datascientest_tia/cours datascientest/MLOPS/Projet/juin24cmlops_rakuten_2/data/preprocessed/X_test_update.csv"
    "images_folder" = "C:/Users/Tia/Documents/datascientest_tia/cours datascientest/MLOPS/Projet/juin24cmlops_rakuten_2/data/preprocessed/image_test"
}

$response = Invoke-RestMethod -Uri "http://localhost:8000/predict/" `
                              -Method POST `
                              -Headers $headers `
                              -Form $form

$response

"""

# Pour mettre à jour les utilisateurs:

"""
# Définir les en-têtes avec le token JWT obtenu lors de l'authentification
$headers = @{
    "Authorization" = "Bearer $token"
    "Content-Type" = "application/json"
}

# Définir le corps de la requête pour mettre à jour le rôle
$body = @{
    "role" = "admin"
} | ConvertTo-Json

# Envoyer la requête PUT pour mettre à jour l'utilisateur
Invoke-RestMethod -Uri "http://127.0.0.1:8000/users/update/" `
                  -Method PUT `
                  -Headers $headers `
                  -Body $body


"""


