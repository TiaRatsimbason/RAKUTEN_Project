#!/bin/bash

# Vérifier si le dossier data existe déjà et contient les sous-dossiers 'raw' et 'preprocessed'
if [ -d "data/raw" ] && [ -d "data/preprocessed" ]; then
    echo "Data directories already exist. Skipping data setup."
else
    # Étape 1 : Préparer les données
    echo "Setting up data..."
    python setup_data.py

    # Vérifier si la préparation des données a réussi
    if [ $? -ne 0 ]; then
        echo "Data setup failed. Exiting."
        exit 1
    fi

    echo "Data setup completed successfully."
fi

# Vérifier si les modèles existent déjà dans le dossier 'models'
if [ -f "models/best_lstm_model.h5" ] && [ -f "models/best_vgg16_model.h5" ] && [ -f "models/concatenate.h5" ]; then
    echo "Models already exist. Skipping model training."
else
    # Étape 2 : Entraîner le modèle
    echo "Training model..."
    python src/main.py

    # Vérifier si l'entraînement du modèle a réussi
    if [ $? -ne 0 ]; then
        echo "Model training failed. Exiting."
        exit 1
    fi

    echo "Model training completed successfully."
fi

# Étape 3 : Démarrer Uvicorn après l'entraînement ou si tout est déjà prêt
echo "Starting API server with Uvicorn..."
uvicorn src.api.app:app --reload
