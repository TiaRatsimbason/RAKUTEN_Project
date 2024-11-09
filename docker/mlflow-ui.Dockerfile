FROM python:3.10.14-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installer MLflow et autres dépendances
RUN pip install --no-cache-dir \
    mlflow==2.8.0 \
    psutil \
    pymongo \
    pandas \
    scikit-learn

# Créer le répertoire de travail
WORKDIR /app

# Créer les répertoires nécessaires
RUN mkdir -p /app/mlruns /app/models && \
    chmod -R 777 /app/mlruns /app/models

# Exposer le port MLflow
EXPOSE 5000

# Définir les variables d'environnement
ENV MLFLOW_SERVE_ARTIFACTS=true
ENV MLFLOW_ENABLE_CORS=true

# Lancer le serveur MLflow
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--serve-artifacts", \
     "--gunicorn-opts", "--timeout 120"]