import mlflow
from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import numpy as np

# Définir l'URI du serveur MLFlow
mlflow.set_tracking_uri("http://mlflow-ui:5000")

# Définir ou récupérer un experiment dans MLFlow
experiment_name = "Rakuten Model Training"
mlflow.set_experiment(experiment_name)

# Démarrer une session MLFlow principale
with mlflow.start_run(run_name="Train Models"):
    data_importer = DataImporter()
    df = data_importer.load_data()
    X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

    # Preprocess text and images
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor()
    text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)

    # Log des paramètres d'entraînement dans MLFlow
    mlflow.log_param("num_classes", 27)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("max_sequence_length", 10)

    # Train LSTM model
    print("Training LSTM Model")
    text_lstm_model = TextLSTMModel()
    text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    print("Finished training LSTM")

    # Train VGG16 model
    print("Training VGG")
    image_vgg16_model = ImageVGG16Model()
    image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    print("Finished training VGG")

    # Train and save the concatenated model
    with mlflow.start_run(run_name="ConcatenateModel", nested=True):
        print("Training the concatenate model")
        tokenizer = text_lstm_model.tokenizer
        lstm = keras.models.load_model("models/best_lstm_model.h5")
        vgg16 = keras.models.load_model("models/best_vgg16_model.h5")
        model_concatenate = concatenate(tokenizer, lstm, vgg16)
        lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
        best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
        print("Finished training concatenate model")

        # Enregistrer les meilleurs poids dans un fichier et dans MLFlow
        with open("models/best_weights.pkl", "wb") as file:
            pickle.dump(best_weights, file)
        mlflow.log_artifact("models/best_weights.pkl")

        # Définir le modèle concaténé avec les poids optimaux
        num_classes = 27
        proba_lstm = keras.layers.Input(shape=(num_classes,))
        proba_vgg16 = keras.layers.Input(shape=(num_classes,))

        weighted_proba = keras.layers.Lambda(
            lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
        )([proba_lstm, proba_vgg16])

        concatenate_model = keras.models.Model(
            inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
        )

        # Enregistrer le modèle concaténé dans MLFlow
        concatenate_model.save("models/concatenate.h5")
        mlflow.tensorflow.log_model(concatenate_model, "ConcatenateModel")

    print("All models trained and saved successfully!")