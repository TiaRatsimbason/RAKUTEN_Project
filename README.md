# Project Name

![Tests](https://github.com/DataScientest-Studio/juin24cmlops_rakuten_2/actions/workflows/test.yml/badge.svg)

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

## Project Organization

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── external
    │   ├── preprocessed
    │   │  ├── image_train
    │   │  └── image_test
    │   └── raw
    │      ├── image_train
    │      └── image_test
    ├── db
    ├── docker
    │   ├── docker-compose
    │   └── dockerfile
    ├── docs
    ├── logs
    │   ├── train
    │   └── validation
    ├── models
    ├── notebooks
    ├── requirements.txt
    ├── src
    │   ├── api
    │   │   ├── root
    │   │   └── utils
    │   └── scripts
    │       ├── data
    │       ├── features
    │       ├── models
    │       └── visualization
    └── tests
        ├── integration
        └── unit
---

Once you have downloaded the GitHub repository and the data (required before running container) following the instructions below, open a WSL terminal at the root of the project folder and you can run the following commands:

> `docker-compose -f docker/docker-compose.yaml build` <- This will build the container images

> `docker-compose -f docker/docker-compose.yaml --env-file .env.dev up -d` <- This will start the containers

> `docker-compose -f docker/docker-compose.yaml down` <- This will stop the containers

> `docker-compose -f docker/docker-compose.yaml down -v` <- This will erase the volumes

| Environnement Windows | Environnement MacOS ou Linux |
|:----------------------|:-----------------------------|
| `pip install -r requirements_win.txt` | `pip install -r requirements_linux_macos.txt` |
| ↳ Installe les packages requis | ↳ Installe les packages requis |


> Download the data folder from Google Drive (https://drive.google.com/drive/home?hl=fr-FR) using these credentials:
>  * Email: projetmlops@gmail.com
>  * Password: MLOps@Rakuten
>  
>  The data folder is located in 'My Drive/MLOps_Rakuten_data/'
> 
>  You need to place the content in the '~/juin24cmlops_rakuten_2/data/preprocessed' folder

    data
    └── preprocessed
        ├── image_test
        ├── image_train
        ├── X_test_update.csv
        ├── X_train_update.csv
        └── Y_train_CvvW08PX.csv
---

> `[http://localhost:8000/docs]` <- You can access the FastAPI documentation at this address once all containers are available and healthy (except for airflow-init)
>
> `[http://localhost:8081/db/rakuten_db/]` <- You can take a look at the content of the database by going to this address and using these credentials: admin:pass
>
> `[http://localhost:5000/]` <- You can take a look at the different training experiments and model versions in the MLflow UI by accessing this URL
>
> Regarding workflow automation, there is Airflow, but to access the web client, you need to create a user with the admin role. Therefore, you need to be able to execute a command in the Airflow container. To do this, you need to execute this command in the WSL terminal: `docker exec -it airflow bash` and then on the command line that opens, you need to enter:
> 
    `airflow users create 
    --username airflow 
    --firstname airflow 
    --lastname airflow 
    --role Admin 
    --email user@airflow.fr 
    --password essai@airflow`
>
> which will give you the credentials to log in into Airflow UI: `airflow:essai@airflow`
>
> `[http://localhost:8080/]` <- You can access the Airflow UI  at this address
>
>##API ENDPOINTS
#1. /load-data/

Checks for required files existence.
Loads data using MongoDBDataLoader (src\scripts\data\mongodb_data_loader.py).
Saves status in the data_pipeline collection on rakuten_db.

#2. /data-status/

Verifies required collections existence.
Counts number of documents in each collection.
Checks for images presence in GridFS.

#3. /prepare-data/

Performs data preprocessing.
Splits data into training, validation and test sets.
Processes images and stores them in GridFS with appropriate metadata (rakuten_db - pipeline_metadata).
Saves labeled data in MongoDB.

#4. /train-model/

Calls train_and_save_model() function from src.scripts.main (rakuten_db - model_metadata).

#5. /predict/

Loads test data from MongoDB.
Loads predictor and performs predictions.
Saves predictions in MongoDB (rakuten_db - predictions collection).

#6. /evaluate-model/

Loads labeled test data.
Makes predictions on this data.
Calculates evaluation metrics.
Saves results in MongoDB (rakuten_db - model_evaluation).

> `python src/scripts/predict.py` <- It will use the trained models to make a prediction (of the prdtypecode) on the desired data, by default, it will predict on the train. You can pass the path to data and images as arguments if you want to change it

    Exemple : python src/scripts/predict_1.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"

                                         The predictions are saved in data/preprocessed as 'predictions.json'

> You can download the trained models loaded here : https://drive.google.com/drive/folders/1fjWd-NKTE-RZxYOOElrkTdOw2fGftf5M?usp=drive_link and insert them in the models folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
python make_dataset.py "../../data/raw" "../../data/preprocessed"

