import datetime
import json
import os

import keras
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score

from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.scripts.models.train_model import TextLSTMModel, ImageVGG16Model, concatenate


def train_and_save_model():
    # Définir l'URI du serveur MLFlow
    mlflow.set_tracking_uri("http://mlflow-ui:5000")

    # Définir l'experiment dans MLFlow
    experiment_name = "Rakuten Model Training"
    mlflow.set_experiment(experiment_name)

    # Générer un timestamp pour rendre chaque run unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Définir une session principale avec `Train Models`
    with mlflow.start_run(run_name=f"Train Models {timestamp}") as main_run:
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

        # Enregistrer le modèle LSTM
        lstm_checkpoint_path = os.path.join("models", "best_lstm_model.keras")
        text_lstm_model.model.save(lstm_checkpoint_path)
        mlflow.log_artifact(lstm_checkpoint_path)

        # Enregistrer le tokenizer LSTM
        tokenizer_path = os.path.join("models", "tokenizer_config.json")
        with open(tokenizer_path, "w", encoding="utf-8") as json_file:
            json_file.write(text_lstm_model.tokenizer.to_json())
        mlflow.log_artifact(tokenizer_path)

        print("Finished training LSTM")

        # Train VGG16 model
        print("Training VGG")
        image_vgg16_model = ImageVGG16Model()
        image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

        # Enregistrer le modèle VGG16
        vgg16_checkpoint_path = os.path.join("models", "best_vgg16_model.keras")
        image_vgg16_model.model.save(vgg16_checkpoint_path)
        mlflow.log_artifact(vgg16_checkpoint_path)
        print("Finished training VGG")

        # Train Concatenate Model
        print("Training the concatenate model")
        tokenizer = text_lstm_model.tokenizer
        lstm = keras.models.load_model(lstm_checkpoint_path)
        vgg16 = keras.models.load_model(vgg16_checkpoint_path)
        model_concatenate = concatenate(tokenizer, lstm, vgg16)
        lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
        best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
        print("Finished training concatenate model")

        # Enregistrer les meilleurs poids dans un fichier
        best_weights_path = os.path.join("models", "best_weights.json")
        with open(best_weights_path, "w") as file:
            json.dump(best_weights, file)
        mlflow.log_artifact(best_weights_path)

        # Définir le modèle concaténé avec les poids optimaux
        print("Defining and saving the concatenated model")
        num_classes = 27
        proba_lstm = keras.layers.Input(shape=(num_classes,))
        proba_vgg16 = keras.layers.Input(shape=(num_classes,))

        weighted_proba = keras.layers.Lambda(
            lambda x, lstm_weight=best_weights[0], vgg16_weight=best_weights[1]:
            lstm_weight * x[0] + vgg16_weight * x[1]
        )([proba_lstm, proba_vgg16])

        concatenate_model = keras.models.Model(
            inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
        )

        # Compiler le modèle concaténé
        concatenate_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Enregistrer le modèle concaténé dans MLFlow
        concatenate_checkpoint_path = os.path.join("models", "concatenate.keras")
        concatenate_model.save(concatenate_checkpoint_path)
        mlflow.log_artifact(concatenate_checkpoint_path)

        # Enregistrer le modèle dans le registre des modèles MLflow
        print("Registering the concatenated model in MLFlow")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts/concatenate.keras"
        mlflow.register_model(model_uri, "RegisteredConcatenateModel")

        # Calculer et enregistrer les métriques du modèle concaténé
        combined_predictions = (best_weights[0] * lstm_proba) + (best_weights[1] * vgg16_proba)
        final_predictions = np.argmax(combined_predictions, axis=1)
        accuracy = accuracy_score(new_y_train, final_predictions)

        # Log the accuracy as a metric
        mlflow.log_metric("concatenate_model_accuracy", accuracy)

    print("All models trained and saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
