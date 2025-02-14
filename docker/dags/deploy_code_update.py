from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os


default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définir le chemin vers docker-compose.yaml
DOCKER_COMPOSE_PATH = os.getenv('DOCKER_COMPOSE_PATH', '/docker/docker-compose.yaml')

with DAG(
    'deploy_code_update_docker',
    default_args=default_args,
    description='DAG pour déployer automatiquement les mises à jour du code des conteneurs Docker',
    schedule_interval=None,  # Le DAG est déclenché manuellement par watchmedo
    start_date=datetime(2023, 1, 1),
    catchup=False, 
) as dag:

    # Arrêter le conteneur existant
    stop_container = BashOperator(
        task_id='stop_container',
        bash_command=f'sudo docker-compose -f {DOCKER_COMPOSE_PATH} stop api'
    )

    # Supprimer le conteneur existant
    remove_container = BashOperator(
        task_id='remove_container',
        bash_command=f'sudo docker-compose -f {DOCKER_COMPOSE_PATH} rm -f api'
    )

    # Démarrer le conteneur en environnement de développement
    start_container = BashOperator(
        task_id='start_containers_dev',
        bash_command=f'sudo docker-compose -f {DOCKER_COMPOSE_PATH} --env-file /docker/.env.dev up -d --no-deps api'
    )


    # Tâche indiquant un déploiement réussi
    deployment_successful = DummyOperator(task_id='deployment_successful')

    # Définir les dépendances mises à jour
    stop_container >> remove_container >> start_container >> deployment_successful
