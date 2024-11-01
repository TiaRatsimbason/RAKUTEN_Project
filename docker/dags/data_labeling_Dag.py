from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from datetime import datetime, timedelta
from build_features import DataImporter, ImagePreprocessor, TextPreprocessor
import os

# Définitions par défaut des arguments du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Fonction Python pour utiliser les classes de build_features.py
def run_build_features():
    # Initialisation des objets
    importer = DataImporter()
    df = importer.load_data()

    # Division des ensembles de données
    X_train, X_val, X_test, y_train, y_val, y_test = importer.split_train_test(df)

    # Prétraitement des images
    image_preprocessor = ImagePreprocessor()
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)
    image_preprocessor.preprocess_images_in_df(X_test)

    # Prétraitement des textes
    text_preprocessor = TextPreprocessor()
    text_preprocessor.preprocess_text_in_df(X_train, ['description'])
    text_preprocessor.preprocess_text_in_df(X_val, ['description'])
    text_preprocessor.preprocess_text_in_df(X_test, ['description'])

    # Sauvegarder les ensembles de données traités
    X_train.to_csv("/opt/airflow/data/processed/X_train.csv", index=False)
    X_val.to_csv("/opt/airflow/data/processed/X_val.csv", index=False)
    X_test.to_csv("/opt/airflow/data/processed/X_test.csv", index=False)
    y_train.to_csv("/opt/airflow/data/processed/y_train.csv", index=False)
    y_val.to_csv("/opt/airflow/data/processed/y_val.csv", index=False)
    y_test.to_csv("/opt/airflow/data/processed/y_test.csv", index=False)

# Définition du DAG
with DAG(
    'data_labeling_pipeline',
    default_args=default_args,
    description='Pipeline pour étiqueter automatiquement les nouveaux points de données',
    schedule_interval=timedelta(days=1),  # Planification quotidienne
    start_date=datetime(2024, 10, 23),
    catchup=False,
    tags=['data_labeling'],
) as dag:

    # Sensor pour surveiller les fichiers dans le bucket S3
    s3_sensor = S3KeySensor(
        task_id='s3_key_sensor',
        bucket_name='mlops-project-db',
        bucket_key='classification_e-commerce/X_train_update.csv',
        aws_conn_id='aws_default',  # L'ID de connexion AWS configuré dans Airflow
        timeout=18 * 60 * 60,
        poke_interval=60 * 10,
        dag=dag,
    )

    # Tâche pour importer les données brutes
    import_raw_data = PythonOperator(
        task_id='import_raw_data',
        python_callable=lambda: os.system('python /opt/airflow/dags/import_raw_data.py'),
    )

    # Tâche pour créer les ensembles de données
    make_dataset = PythonOperator(
        task_id='make_dataset',
        python_callable=lambda: os.system('python /opt/airflow/dags/make_dataset.py'),
    )

    # Tâche pour construire les features
    build_features = PythonOperator(
        task_id='build_features',
        python_callable=run_build_features,
    )

    # Définition de la séquence des tâches
    s3_sensor >> import_raw_data >> make_dataset >> build_features
