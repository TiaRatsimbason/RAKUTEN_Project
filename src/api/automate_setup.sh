#!/bin/bash

# Étape 0 : Créer l'environnement conda s'il n'existe pas
ENV_NAME="Rakuten-project"

# Vérifier si l'environnement conda existe déjà
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Activating it..."
    # Activer l'environnement conda
    source activate $ENV_NAME
else
    # Créer l'environnement conda
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.10.14 -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment. Exiting."
        exit 1
    fi
    echo "Conda environment '$ENV_NAME' created successfully."

    # Activer l'environnement conda
    source activate $ENV_NAME

    # Installer pip et mettre à jour pip
    echo "Installing pip and updating it..."
    conda install pip -y
    python -m pip install -U pip
    if [ $? -ne 0 ]; then
        echo "Failed to install or update pip. Exiting."
        exit 1
    fi

    # Installer les dépendances via requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install dependencies. Exiting."
            exit 1
        fi
        echo "Dependencies installed successfully."
    else
        echo "No requirements.txt file found. Skipping dependency installation."
    fi
fi

# Vérifier les dépendances manquantes
echo "Checking for missing dependencies..."
python -m pip install --upgrade pip  # Mettre à jour pip avant de vérifier les dépendances manquantes

MISSING_DEPENDENCIES=$(pip check 2>&1 | grep -i "No module named" | awk '{print $5}')
if [ ! -z "$MISSING_DEPENDENCIES" ]; then
    for dep in $MISSING_DEPENDENCIES; do
        echo "Installing missing dependency: $dep"
        pip install $dep
        if [ $? -ne 0 ]; then
            echo "Failed to install dependency: $dep. Exiting."
            exit 1
        fi
    done
else
    echo "All dependencies are satisfied."
fi

# Étape 1 : Préparer les données
if [ -d "data/raw" ] && [ -d "data/preprocessed" ]; then
    echo "Data directories already exist. Skipping data setup."
else
    echo "Setting up data..."
    python src/scripts/data/setup_data_cloud.py
    if [ $? -ne 0 ]; then
        echo "Data setup failed. Exiting."
        exit 1
    fi
    echo "Data setup completed successfully."
fi

# Étape 2 : Vérifier si les modèles existent déjà dans le dossier 'models'
if [ -f "models/best_lstm_model.h5" ] && [ -f "models/best_vgg16_model.h5" ] && [ -f "models/concatenate.h5" ]; then
    echo "Models already exist. Skipping model training."
else
    echo "Training model..."
    python src/scripts/main.py
    if [ $? -ne 0 ]; then
        echo "Model training failed. Exiting."
        exit 1
    fi
    echo "Model training completed successfully."
fi

# Étape 3 : Remplacer les modèles entraînés par ceux de Google Drive, le cas échéant
echo "Checking for models on Google Drive..."

# Fonction pour détecter le chemin Google Drive
detect_google_drive_path() {
    drives=$(echo {A..Z})
    for drive in $drives; do
        if [ -d "$drive:/Mon Drive" ]; then
            echo "$drive:/Mon Drive"
            return 0
        fi
    done
    return 1
}

google_drive_path=$(detect_google_drive_path)
if [ $? -eq 0 ]; then
    models_drive_path="$google_drive_path/models"
    
    if [ -d "$models_drive_path" ]; then
        echo "Replacing trained models with models from Google Drive..."

        # Remplacer les modèles locaux par ceux du Google Drive
        cp -f "$models_drive_path/best_lstm_model.h5" "models/best_lstm_model.h5"
        cp -f "$models_drive_path/best_vgg16_model.h5" "models/best_vgg16_model.h5"

        echo "Models replaced with those from Google Drive."
    else
        echo "No models found on Google Drive. Keeping trained models."
    fi
else
    echo "Google Drive not found. Keeping trained models."
fi

# Étape 4 : Démarrer Uvicorn après l'entraînement ou si tout est déjà prêt
echo "Starting API server with Uvicorn..."
uvicorn src.api.app:app --reload
