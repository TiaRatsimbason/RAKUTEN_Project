from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests
import os

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définir le jeton et d'autres informations
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Utilisez une variable d'environnement pour la sécurité
REPO_OWNER = "DataScientest-Studio"  # Remplacez par l'utilisateur ou l'organisation du dépôt
REPO_NAME = "juin24cmlops_rakuten_2"  # Nom du dépôt
BRANCH = "Dev"  # Branche que vous souhaitez surveiller

def check_for_updates(**kwargs):
    if not GITHUB_TOKEN:
        raise ValueError("Le GITHUB_TOKEN n'est pas défini dans l'environnement.")

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/branches/{BRANCH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        branch_info = response.json()
        latest_commit_sha = branch_info['commit']['sha']

        # Stocker ou comparer avec le dernier commit connu
        if not os.path.exists('/path/to/last_commit_sha.txt'):
            with open('/path/to/last_commit_sha.txt', 'w') as f:
                f.write('')  # Crée le fichier s'il n'existe pas encore

        with open('/path/to/last_commit_sha.txt', 'r+') as f:
            last_commit_sha = f.read().strip()

            is_update_needed = last_commit_sha != latest_commit_sha
            if is_update_needed:
                # Mettez à jour le fichier avec le nouveau SHA
                f.seek(0)
                f.write(latest_commit_sha)
                f.truncate()

            return is_update_needed
    else:
        raise Exception(f"Failed to fetch branch info: {response.status_code}, {response.text}")

def branch_check_for_updates(**kwargs):
    if kwargs['task_instance'].xcom_pull(task_ids='check_for_updates'):
        return 'stop_and_remove_containers'
    else:
        return 'no_update_needed'

def check_containers_health(**kwargs):
    import subprocess
    result = subprocess.run("docker ps --filter 'status=exited' --format '{{.Names}}'", shell=True, capture_output=True, text=True)
    if result.stdout:
        # Un ou plusieurs conteneurs sont en état 'exited'
        return 'rollback_to_previous_version'
    return 'deployment_successful'

with DAG(
    'deploy_code_update_docker',
    default_args=default_args,
    description='DAG pour déployer automatiquement les mises à jour du code des conteneurs Docker',
    schedule_interval=timedelta(hours=1),  # exécuter toutes les heures
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Vérifier les mises à jour dans le dépôt
    detect_updates = PythonOperator(
        task_id='check_for_updates',
        python_callable=check_for_updates,
        provide_context=True
    )

    # Brancher la logique pour déterminer si une mise à jour est nécessaire
    branch_task = BranchPythonOperator(
        task_id='branch_check_for_updates',
        python_callable=branch_check_for_updates,
        provide_context=True
    )

    # Tâche pour quand aucune mise à jour n'est nécessaire
    no_update_needed = DummyOperator(task_id='no_update_needed')

    # Arrêter et supprimer les conteneurs existants
    stop_and_remove_containers = BashOperator(
        task_id='stop_and_remove_containers',
        bash_command='docker-compose -f docker/docker-compose.yaml down'
    )

    # Construire les images Docker
    build_images = BashOperator(
        task_id='build_docker_images',
        bash_command='docker-compose -f docker/docker-compose.yaml build'
    )

    # Démarrer les conteneurs en environnement de développement
    start_containers = BashOperator(
        task_id='start_containers_dev',
        bash_command='docker-compose -f docker/docker-compose.yaml --env-file .env.dev up -d'
    )

    # Vérifier la santé des conteneurs (sauf airflow-init)
    check_health = PythonOperator(
        task_id='check_containers_health',
        python_callable=check_containers_health,
        provide_context=True
    )

    # Tâche pour rollback si un conteneur est en état 'exited'
    rollback_to_previous_version = BashOperator(
        task_id='rollback_to_previous_version',
        bash_command='docker-compose -f docker/docker-compose.yaml down && git checkout HEAD~1 && docker-compose -f docker/docker-compose.yaml build && docker-compose -f docker/docker-compose.yaml --env-file .env.dev up -d'
    )

    # Tâche indiquant un déploiement réussi
    deployment_successful = DummyOperator(task_id='deployment_successful')

    # Définir les dépendances mises à jour
    detect_updates >> branch_task
    branch_task >> [stop_and_remove_containers, no_update_needed]
    stop_and_remove_containers >> build_images >> start_containers >> check_health
    check_health >> [rollback_to_previous_version, deployment_successful]
