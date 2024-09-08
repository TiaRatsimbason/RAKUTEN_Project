# Project Name

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources -> the external data you want to make a prediction on
    │   ├── preprocessed      <- The final, canonical data sets for modeling.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    |   |  ├── X_train_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    |   |  ├── X_test_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    │   └── raw            <- The original, immutable data dump.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- Scripts to train models
    │   ├── predict.py     <- Scripts to use trained models to make prediction on the files put in ../data/preprocessed
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py
    │   │   ├── import_raw_data.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models
    │   │   └── train_model.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

---

Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project" python==3.10.14` <- It will create your conda environement with python 3.10.14

> `conda activate Rakuten-project` <- It will activate your environment

> `conda install pip` <- May be optionnal

> `python -m pip install -U pip` <- Upgrade to the latest available version of pip

> `pip install -r requirements.txt` <- It will install the required packages

> `python src/data/import_raw_data.py` <- It will import the tabular data on data/raw/

> Upload the image data folder set directly on local from https://challengedata.ens.fr/participants/challenges/35/, you should save the folders image_train and image_test respecting the following structure

    ├── data
    │   └── raw
    |   |  ├── image_train
    |   |  ├── image_test

> `python src/data/make_dataset.py data/raw data/preprocessed` <- It will copy the raw dataset and paste it on data/preprocessed/

> `python src/main.py` <- It will train the models on the dataset and save them in models. By default, the number of epochs = 1

> `python src/predict.py` <- It will use the trained models to make a prediction (of the prdtypecode) on the desired data, by default, it will predict on the train. You can pass the path to data and images as arguments if you want to change it

    Exemple : python src/predict_1.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"

                                         The predictions are saved in data/preprocessed as 'predictions.json'

> You can download the trained models loaded here : https://drive.google.com/drive/folders/1fjWd-NKTE-RZxYOOElrkTdOw2fGftf5M?usp=drive_link and insert them in the models folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
python make_dataset.py "../../data/raw" "../../data/preprocessed"

**Commandes pour lancer l'API**

## Commandes Bash (il faut un terminal avec commandes bash)

## Pour executer l'API:

"""
./src/api/automate_setup.sh
"""

## Pour charger les données en local (si les données ont été mises à jour):

"""
curl -X POST "http://localhost:8000/setup-data/"
"""

## Pour enregistrer un utilisateur:

"""
headers="Content-Type: application/json"
body='{
"username": "Tia",
"password": "Tia@7777",
"role": "user"
}'
response=$(curl -X POST "http://127.0.0.1:8000/users/register/" \
    -H "$headers" \
 -d "$body")
echo "Response: $response"

"""

## Pour avoir l'access_token:

"""

# Définir les en-têtes

headers="Content-Type: application/x-www-form-urlencoded"

# Définir le corps de la requête

body="username=Tia&password=Tia@7777"

# Faire une requête HTTP POST avec curl

response=$(curl -X POST "http://localhost:8000/users/token" \
    -H "$headers" \
 --data "$body")

# Extraire le token de la réponse

token=$(echo $response | jq -r '.access_token')

# Afficher le token (si `jq` est installé pour traiter les JSON)

echo "Token: $token"

"""

## Pour entraîner le model:

"""
curl -X POST "http://localhost:8000/train-model/" -H "Authorization: Bearer $token"

"""

## Pour faire une requête à l'api:

"""
headers="Authorization: Bearer $token"
file_path="C:\Users\Tia\Documents\projet RAKUTEN MLOps\juin24cmlops_rakuten_2\data\preprocessed/X_test_update.csv"
##remplacer par le bon chemin sur votre PC
images_folder="C:\Users\Tia\Documents\projet RAKUTEN MLOps\juin24cmlops_rakuten_2\data\preprocessed/image_test"
##remplacer par le bon chemin sur votre PC
response=$(curl -X POST "http://localhost:8000/predict/" \
 -H "Content-Type: multipart/form-data" \
 -H "$headers" \
    -F "file=@$file_path" \
 -F "images_folder=$images_folder")
echo "Response: $response"
"""

## Pour mettre à jour un utilisateur:

"""

headers="Authorization: Bearer $token"
body='{"role": "admin"}'
response=$(curl -X PUT "http://127.0.0.1:8000/users/update/" \
 -H "$headers" \
    -H "Content-Type: application/json" \
    -d "$body")
echo "Response: $response"

"""
