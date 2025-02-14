FROM apache/airflow:2.5.1-python3.10

USER root

# Mis à jour de pip
RUN python -m pip install --upgrade pip

# Installation de Docker Compose
RUN curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

# Étape existante pour ajouter l'utilisateur au groupe Docker
RUN groupadd docker && usermod -aG docker airflow

# Ajoutez la configuration sudoers pour l'utilisateur airflow
RUN echo "airflow ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/local/bin/docker-compose" >> /etc/sudoers

# Passer à l'utilisateur airflow
USER airflow

# Définition du répertoire de travail
WORKDIR /opt/airflow

# Installation dépendances pour les DAGs
RUN python -m pip install --user nltk && python -m nltk.downloader punkt