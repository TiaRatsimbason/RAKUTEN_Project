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
        libhdf5-serial-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copier seulement les fichiers nécessaires à l'installation des dépendances
COPY pyproject.toml poetry.lock* ./

# Installer Poetry et les dépendances de l'application
# Ce `RUN` installe Poetry, configure pour éviter le venv, installe les paquets, et nettoie le cache
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi && \
    rm -rf ~/.cache/pip

# Copier le reste du code de l’application
COPY src/ /app/src/
COPY models /app/models/
COPY data/container/* /app/data/preprocessed/

# Télécharger les ressources NLTK nécessaires
RUN python -m nltk.downloader punkt_tab

# Exposer le port 8000
EXPOSE 8000

# Add the app directory to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run the FastAPI server
CMD ["poetry", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
