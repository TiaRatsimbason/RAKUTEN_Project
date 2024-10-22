import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import os
import logging

BATCH_SIZE = 23
NUM_CLASSES = 27
ARTIFACTS_DIR = "./models"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Log hyperparameters if not already logged
        current_run = mlflow.active_run()
        if current_run and not mlflow.get_run(current_run.info.run_id).data.params:
            mlflow.log_param("max_words", self.max_words)
            mlflow.log_param("max_sequence_length", self.max_sequence_length)
            mlflow.log_param("batch_size", BATCH_SIZE)

        # Tokenizer configuration
        self.tokenizer.fit_on_texts(X_train["description"])
        tokenizer_config = self.tokenizer.to_json()
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        tokenizer_path = os.path.join(ARTIFACTS_DIR, "tokenizer_config.json")
        with open(tokenizer_path, "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)
        # Log avant l'enregistrement du tokenizer
        logger.info("Logging tokenizer_config.json to MLFlow")
        mlflow.log_artifact(tokenizer_path)

        # Prepare sequences
        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded_sequences = pad_sequences(train_sequences, maxlen=self.max_sequence_length, padding="post", truncating="post")
        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded_sequences = pad_sequences(val_sequences, maxlen=self.max_sequence_length, padding="post", truncating="post")

        # LSTM Model
        text_input = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(text_input)
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(NUM_CLASSES, activation="softmax")(lstm_layer)

        self.model = Model(inputs=[text_input], outputs=output)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # LSTM Model checkpoint path
        lstm_checkpoint_path = os.path.join(ARTIFACTS_DIR, "best_lstm_model.h5")
        lstm_callbacks = [
            ModelCheckpoint(filepath=lstm_checkpoint_path, save_best_only=True),
            EarlyStopping(patience=3, restore_best_weights=True),
            TensorBoard(log_dir="logs"),
        ]

        # Train the model
        history = self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES),
            epochs=1,
            batch_size=BATCH_SIZE,
            validation_data=([val_padded_sequences], tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)),
            callbacks=lstm_callbacks,
        )

        # Log metrics to MLFlow
        mlflow.log_metric("lstm_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("lstm_val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("lstm_loss", history.history['loss'][-1])
        mlflow.log_metric("lstm_val_loss", history.history['val_loss'][-1])

        # Log avant l'enregistrement du modèle LSTM
        logger.info("Logging LSTM model to MLFlow")
        # Log the LSTM model
        mlflow.tensorflow.log_model(self.model, "TextLSTMModel")
        mlflow.log_artifact("logs")


class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Log hyperparameters if not already logged
        current_run = mlflow.active_run()
        if current_run and not mlflow.get_run(current_run.info.run_id).data.params:
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("num_classes", NUM_CLASSES)

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Image Data Generators
        train_datagen = ImageDataGenerator()
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )

        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        # VGG16 Model
        image_input = Input(shape=(224, 224, 3))
        vgg16_base = VGG16(include_top=False, weights="imagenet", input_tensor=image_input)
        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        output = Dense(NUM_CLASSES, activation="softmax")(x)

        self.model = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        vgg_checkpoint_path = os.path.join(ARTIFACTS_DIR, "best_vgg16_model.h5")
        vgg_callbacks = [
            ModelCheckpoint(filepath=vgg_checkpoint_path, save_best_only=True),
            EarlyStopping(patience=3, restore_best_weights=True),
            TensorBoard(log_dir="logs"),
        ]

        # Train the model
        history = self.model.fit(train_generator, epochs=1, validation_data=val_generator, callbacks=vgg_callbacks)

        # Log metrics
        mlflow.log_metric("vgg16_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("vgg16_val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("vgg16_loss", history.history['loss'][-1])
        mlflow.log_metric("vgg16_val_loss", history.history['val_loss'][-1])

        # Log avant l'enregistrement du modèle VGG16
        logger.info("Logging VGG16 model to MLFlow")
        # Log the VGG16 model
        mlflow.tensorflow.log_model(self.model, "ImageVGG16Model")
        mlflow.log_artifact("logs")


class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10):
        num_classes = NUM_CLASSES

        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.DataFrame(columns=[])

        # Resample data for each class
        for class_label in range(num_classes):
            indices = np.where(y_train == class_label)[0]
            sampled_indices = resample(indices, n_samples=new_samples_per_class, replace=False, random_state=42)
            new_X_train = pd.concat([new_X_train, X_train.loc[sampled_indices]])
            new_y_train = pd.concat([new_y_train, y_train.loc[sampled_indices]])

        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)
        new_y_train = new_y_train.values.reshape(-1).astype("int")

        train_sequences = self.tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding="post", truncating="post")

        # Preprocess images
        target_size = (224, 224, 3)
        images_train = new_X_train["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        # Make predictions using LSTM and VGG16
        lstm_proba = self.lstm.predict([train_padded_sequences])
        vgg16_proba = self.vgg16.predict([images_train])

        return lstm_proba, vgg16_proba, new_y_train

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        best_weights = None
        best_accuracy = 0.0

        # Find optimal weights for combining LSTM and VGG16 predictions
        for lstm_weight in np.linspace(0, 1, 101):
            vgg16_weight = 1.0 - lstm_weight
            combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        # Enregistrer les meilleurs poids dans MLflow
        logger.info("Logging best weights to MLFlow")
        mlflow.log_param("best_lstm_weight", best_weights[0])
        mlflow.log_param("best_vgg16_weight", best_weights[1])

        return best_weights
