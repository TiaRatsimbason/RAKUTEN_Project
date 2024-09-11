# Project Name

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
    │   ├── data
    │   ├── features
    │   ├── models
    │   └── config
    └── tests
        ├── integration
        └── unit
---

Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project" python==3.10.14` <- It will create your conda environement with python 3.10.14

> `conda activate Rakuten-project` <- It will activate your environment

> `conda install pip` <- May be optionnal

> `python -m pip install -U pip` <- Upgrade to the latest available version of pip

| Environnement Windows | Environnement MacOS ou Linux |
|:----------------------|:-----------------------------|
| `pip install -r requirements_win.txt` | `pip install -r requirements_linux_macos.txt` |
| ↳ Installe les packages requis | ↳ Installe les packages requis |

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

