FROM apache/airflow:2.5.1-python3.10

USER root

# Mis à jour de pip
RUN python -m pip install --upgrade pip

# Passer à l'utilisateur airflow
USER airflow

# Définition du répertoire de travail
WORKDIR /opt/airflow

# Installation dépendances pour les DAGs
RUN python -m pip install --user nltk && python -m nltk.downloader punkt