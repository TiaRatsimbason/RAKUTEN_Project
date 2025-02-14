# RAKUTEN_Project
1. Clone the repository.
2. Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project" python==3.10.14` <- It will create your conda environement with python 3.10.14

> `conda activate Rakuten-project` <- It will activate your environment

> `conda install pip` <- May be optionnal

> `python -m pip install -U pip` <- Upgrade to the latest available version of pip

| Environnement Windows | Environnement MacOS ou Linux |
|:----------------------|:-----------------------------|
| `pip install -r requirements_win.txt` | `pip install -r requirements_linux_macos.txt` |
| ↳ Installe les packages requis | ↳ Installe les packages requis |

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

## Prerequisites
Docker
## Installation
1. Download the data from Google Drive.
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

## Usage

1. Build the Docker images:
   ```bash
   docker-compose -f docker/docker-compose.yaml build
   
2. Start the containers:
   ```bash
   docker-compose -f docker/docker-compose.yaml --env-file .env.dev up -d

3. Stop the containers:
   ```bash
   docker-compose -f docker/docker-compose.yaml down

4. Erase the volumes:
   ```bash
   docker-compose -f docker/docker-compose.yaml down -v

## WebClients

* You can access the FastAPI documentation at this address once all containers are available and healthy (except for `airflow-init`):
   ```bash
   http://localhost:8000/docs
    ```
* You can take a look at the content of the database by going to this address:
   ```bash
   http://localhost:8081/db/rakuten_db/
    ```
* You can take a look at the different training experiments and model versions in the MLflow UI by going to:
   ```bash
   http://localhost:5000/
    ```
Regarding workflow automation, there is Airflow, but to access the web client, you need to create a user with the admin role. Therefore, you need to be able to execute a command in the Airflow container. To do this, you need to execute this command in the WSL terminal:

   ```bash
   docker exec -it airflow bash
   ```
and then on the command line that opens, you need to enter:

```bash
airflow users create \
  --username airflow \
  --firstname airflow \
  --lastname airflow \
  --role Admin \
  --email user@airflow.fr \
  --password essai@airflow
 ```


 which will give you the credentials to log in into Airflow UI: `airflow:essai@airflow`

* You can access the Airflow UI  at this address :
   ```bash
   http://localhost:8080/
    ```
## API ENDPOINTS

### 1. /load-data/
* Checks for required files existence.
* Loads data using MongoDBDataLoader (`src\scripts\data\mongodb_data_loader.py`).
* Saves status in the data_pipeline collection on `rakuten_db`.

### 2. /data-status/
* Verifies required collections existence.
* Counts number of documents in each collection.
* Checks for images presence in GridFS.

### 3. /prepare-data/
* Performs data preprocessing.
* Splits data into training, validation and test sets.
* Processes images and stores them in GridFS with appropriate metadata (`rakuten_db` - `pipeline_metadata`).
* Saves labeled data in MongoDB.

### 4. /train-model/
* Calls train_and_save_model() function from src.scripts.main (`rakuten_db` - `model_metadata`).

### 5. /predict/
* Loads test data from MongoDB.
* Loads predictor and performs predictions.
* Saves predictions in MongoDB (`rakuten_db` - `predictions collection`).

### 6. /evaluate-model/
* Loads labeled test data.
* Makes predictions on this data.
* Calculates evaluation metrics.
* Saves results in MongoDB (`rakuten_db` - `model_evaluation`).

## Rakuten Project Database Collections

By executing all API endpoints, collections will be created gradually in the `rakuten_db` MongoDB database. Below are descriptions of all these collections:

The Rakuten project database `rakuten_db` contains several collections that are used throughout the data processing, training, evaluation, and prediction stages of our machine learning pipeline. Below is a detailed description of each collection in rakuten_db and its purpose:

### 1. data_pipeline
This collection stores metadata related to the data pipeline process. It keeps track of the execution status, the files processed, start and end times, and any errors or warnings that occurred during the data loading or preprocessing stages.

### 2. fs.chunks and fs.files
These collections are part of MongoDB's GridFS, used to store large files, such as images. GridFS breaks files into smaller chunks to store them efficiently.
- **fs.files**: Contains metadata about each stored file, such as the original path and information used to retrieve the image.
- **fs.chunks**: Contains the actual chunks of the files that are split by GridFS for storage.

### 3. labeled_test, labeled_train, labeled_val
These collections contain the labeled data used for model training, validation, and testing. Each document in these collections includes features like product descriptions, image identifiers, and associated labels.
- **labeled_train**: Stores the training dataset with the labeled records.
- **labeled_val**: Contains labeled data for model validation.
- **labeled_test**: Stores the labeled test data used to evaluate model performance.

### 4. model_evaluation
This collection contains the evaluation results of different versions of the model. It stores evaluation metrics such as precision, recall, F1-score, inference times, and the version number of the model being evaluated. It helps in tracking model performance over time.

### 5. model_metadata
This collection stores metadata related to each trained model version. It includes details such as the training date, the MLflow run ID, model version, training metrics, and data distribution during training. It helps in keeping track of the different versions of the model and their respective training contexts.

### 6. pipeline_metadata
This collection contains information about the entire data processing pipeline. It includes metadata about the pipeline's execution status, records processed, image processing statistics, and the overall distribution of the dataset across training, validation, and test splits.

### 7. predictions
This collection stores the predictions made by the model on the test data. Each entry contains details such as the model version used, prediction date, and predicted values. This collection helps in analyzing how the model performs on real-world data after training.

### 8. preprocessed_x_train, preprocessed_x_test, preprocessed_y_train
These collections store the preprocessed versions of the training and test datasets.
- **preprocessed_x_train**: Contains the preprocessed feature data for training.
- **preprocessed_x_test**: Contains the preprocessed feature data for testing.
- **preprocessed_y_train**: Contains the preprocessed labels for training.

The collections `labeled_*` and `preprocessed_*` are used to handle different stages of the data lifecycle, ensuring that each phase of the machine learning pipeline has a dedicated storage structure. This organization helps maintain a clear distinction between raw and labeled data, making it easier to manage and analyze data throughout the project lifecycle.

## Contributing

We are excited that you want to contribute to this project! Here are some steps to help you get started.

## How to Contribute

1. Fork the project: Click on the "Fork" button at the top of the repository page.

2. Clone your fork: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your_username/juin24cmlops_rakuten2.git

3. Create a branch: Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b my_new_feature
4. Make your changes: Make your changes in your preferred code editor.
5. Commit your changes: Commit your changes with a clear and descriptive commit message.
   ```bash
   git commit -m "Add my new feature"
6. Push to your fork: Push the changes to your fork on GitHub.
   ```bash
   git commit -m "Add my new feature"
7. Create a Pull Request: Go to the original repository and create a Pull Request from your fork.

## Contribution Guidelines

* Ensure your code follows the best coding practices.
* Add unit tests for new features or bug fixes.
* Update documentation if necessary.
* Clearly describe the changes in your Pull Request.

## License


> You can download the trained models loaded here : https://drive.google.com/drive/folders/1fjWd-NKTE-RZxYOOElrkTdOw2fGftf5M?usp=drive_link and insert them in the models folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
python make_dataset.py "../../data/raw" "../../data/preprocessed"
