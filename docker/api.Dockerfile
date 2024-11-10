# Utiliser l'image Python Slim comme base
FROM python:3.10.14-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires dans une seule commande RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pandoc \
        pkg-config \
        libhdf5-serial-dev \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copier seulement les fichiers nécessaires à l'installation des dépendances
COPY pyproject.toml poetry.lock* ./

# Installer Poetry et les dépendances de l'application
# Ce `RUN` installe Poetry, configure pour éviter le venv, installe les paquets, et nettoie le cache
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi && \
    rm -rf ~/.cache/pip
    
# Créer les répertoires nécessaires
RUN mkdir -p /app/models /app/mlruns /app/data/preprocessed/image_train /app/data/preprocessed/image_test /app/logs

# Copier le reste du code de l’application
#COPY src/ /app/src/
#COPY models/* /app/models/
#COPY data/container/*.csv /app/data/preprocessed/
#COPY data/container/image_train/* /app/data/preprocessed/image_train/
#COPY data/container/image_test/* /app/data/preprocessed/image_test/


# Copier le reste du code de l’application
COPY src/ /app/src/
COPY models /app/models/
COPY data/preprocessed/ /app/data/preprocessed/

# Télécharger les ressources NLTK nécessaires
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Exposer le port 8000
EXPOSE 8000

# Add the app directory to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run the FastAPI server
CMD ["poetry", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

