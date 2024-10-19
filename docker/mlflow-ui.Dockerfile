# Utiliser une image Python basée sur Debian
FROM python:3.10.14-slim-bullseye

# Installer les dépendances système nécessaires pour MLFlow et pyarrow
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    curl \
    git \
    cmake \
    && apt-get clean

# Mettre à jour pip à la dernière version
RUN pip install --upgrade pip

# Installer MLFlow et pyarrow
RUN pip install --no-cache-dir mlflow pyarrow

# Créer les volumes nécessaires pour les artefacts MLFlow
VOLUME ["/mlruns"]

# Exposer le port 5000 pour l'interface MLFlow
EXPOSE 5000

# Lancer le serveur MLFlow
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlruns"]
