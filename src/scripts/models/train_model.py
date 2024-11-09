# Standard library imports
import json
import os
import logging
from datetime import datetime
import io

# Third-party imports
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
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
from bson import ObjectId
from src.config.mongodb import sync_db, async_db, sync_fs, async_fs

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
        # Tokenizer configuration
        self.tokenizer.fit_on_texts(X_train["description"])
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        # Save tokenizer config
        tokenizer_path = os.path.join(ARTIFACTS_DIR, "tokenizer_config.json")
        with open(tokenizer_path, "w", encoding="utf-8") as json_file:
            json_file.write(self.tokenizer.to_json())
        mlflow.log_artifact(tokenizer_path)

        # Prepare sequences
        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded = pad_sequences(train_sequences, maxlen=self.max_sequence_length)
        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded = pad_sequences(val_sequences, maxlen=self.max_sequence_length)

        # Create and train model
        text_input = Input(shape=(self.max_sequence_length,))
        x = Embedding(input_dim=self.max_words, output_dim=128)(text_input)
        x = LSTM(128)(x)
        output = Dense(NUM_CLASSES, activation="softmax")(x)

        self.model = Model(inputs=[text_input], outputs=output)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Callbacks
        callbacks = [
            ModelCheckpoint(os.path.join(ARTIFACTS_DIR, "best_lstm_model.keras"), save_best_only=True),
            EarlyStopping(patience=3, restore_best_weights=True),
            TensorBoard(log_dir="logs")
        ]

        # Train
        history = self.model.fit(
            train_padded,
            tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
            epochs=10,
            batch_size=BATCH_SIZE,
            validation_data=(val_padded, tf.keras.utils.to_categorical(y_val, NUM_CLASSES)),
            callbacks=callbacks
        )

        # Log metrics
        for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            mlflow.log_metric(f"lstm_{metric}", history.history[metric][-1])

class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def create_gridfs_generator(self, df, labels, datagen, split_name):
        target_size = (224, 224)
        
        def generate_batches():
            while True:
                # Shuffle indices
                indices = np.random.permutation(len(df))
                for start in range(0, len(df), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(df))
                    batch_indices = indices[start:end]
                    
                    # Initialize batch arrays
                    batch_images = np.zeros((len(batch_indices), *target_size, 3))
                    batch_labels = np.zeros((len(batch_indices), NUM_CLASSES))
                    
                    # Load images from GridFS
                    for i, idx in enumerate(batch_indices):
                        try:
                            # Get image from GridFS
                            grid_out = sync_fs.get(ObjectId(df.iloc[idx]['gridfs_file_id']))
                            img_data = grid_out.read()
                            
                            # Convert to image and preprocess
                            img = Image.open(io.BytesIO(img_data))
                            img = img.resize(target_size)
                            img_array = img_to_array(img)
                            img_array = preprocess_input(img_array)
                            
                            batch_images[i] = img_array
                            batch_labels[i] = tf.keras.utils.to_categorical(
                                labels.iloc[idx], NUM_CLASSES
                            )
                            
                        except Exception as e:
                            logger.error(f"Error loading image {df.iloc[idx]['gridfs_file_id']}: {e}")
                            continue
                    
                    yield batch_images, batch_labels
        
        return generate_batches()

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Create data generators
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = self.create_gridfs_generator(X_train, y_train, train_datagen, "train")
        val_generator = self.create_gridfs_generator(X_val, y_val, val_datagen, "validation")

        # Create and compile model
        base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        x = Flatten()(base_model.output)
        x = Dense(256, activation="relu")(x)
        output = Dense(NUM_CLASSES, activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=output)
        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Callbacks
        callbacks = [
            ModelCheckpoint(os.path.join(ARTIFACTS_DIR, "best_vgg16_model.keras"), save_best_only=True),
            EarlyStopping(patience=3, restore_best_weights=True),
            TensorBoard(log_dir="logs")
        ]

        # Train
        steps_per_epoch = len(X_train) // BATCH_SIZE
        validation_steps = len(X_val) // BATCH_SIZE

        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        # Log metrics
        for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            mlflow.log_metric(f"vgg16_{metric}", history.history[metric][-1])

class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def predict(self, X_train, y_train, new_samples_per_class=50):
        logger.info("Starting concatenate model prediction...")
        
        # Resample data
        resampled_indices = []
        for class_label in range(NUM_CLASSES):
            class_indices = np.where(y_train == class_label)[0]
            if len(class_indices) >= new_samples_per_class:
                sampled = np.random.choice(class_indices, new_samples_per_class, replace=False)
            else:
                sampled = np.random.choice(class_indices, new_samples_per_class, replace=True)
            resampled_indices.extend(sampled)

        X_resampled = X_train.iloc[resampled_indices]
        y_resampled = y_train.iloc[resampled_indices]

        # Process text
        sequences = self.tokenizer.texts_to_sequences(X_resampled["description"])
        padded_sequences = pad_sequences(sequences, maxlen=10)
        
        # Process images
        images = np.zeros((len(X_resampled), 224, 224, 3))
        for i, (_, row) in enumerate(X_resampled.iterrows()):
            try:
                grid_out = sync_fs.get(ObjectId(row['gridfs_file_id']))
                img_data = grid_out.read()
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                images[i] = img_array
            except Exception as e:
                logger.error(f"Error processing image {row['gridfs_file_id']}: {e}")
                continue

        # Get predictions
        lstm_proba = self.lstm.predict(padded_sequences)
        vgg16_proba = self.vgg16.predict(images)

        return lstm_proba, vgg16_proba, y_resampled

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        best_weights = None
        best_accuracy = 0.0

        for lstm_weight in np.linspace(0, 1, 101):
            vgg16_weight = 1.0 - lstm_weight
            combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
            predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        logger.info(f"Best weights - LSTM: {best_weights[0]:.3f}, VGG16: {best_weights[1]:.3f}")
        logger.info(f"Best accuracy: {best_accuracy:.3f}")

        mlflow.log_param("best_lstm_weight", best_weights[0])
        mlflow.log_param("best_vgg16_weight", best_weights[1])
        mlflow.log_metric("best_combined_accuracy", best_accuracy)

        return best_weights, best_accuracy

def train_and_save_models(X_train, y_train, X_val, y_val):
    try:
        # 1. Train LSTM
        logger.info("Training LSTM Model...")
        text_lstm_model = TextLSTMModel()
        text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

        lstm_path = os.path.join(ARTIFACTS_DIR, "best_lstm_model.keras")
        text_lstm_model.model.save(lstm_path)
        mlflow.log_artifact(lstm_path)

        # 2. Train VGG16
        logger.info("Training VGG16 Model...")
        image_vgg16_model = ImageVGG16Model()
        image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

        vgg16_path = os.path.join(ARTIFACTS_DIR, "best_vgg16_model.keras")
        image_vgg16_model.model.save(vgg16_path)
        mlflow.log_artifact(vgg16_path)

        # 3. Train concatenate Model
        logger.info("Training concatenate model...")
        model_concatenate = concatenate(text_lstm_model.tokenizer, 
                                      text_lstm_model.model, 
                                      image_vgg16_model.model)
        
        lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
        best_weights, best_accuracy = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)

        # Save weights
        weights_path = os.path.join(ARTIFACTS_DIR, "best_weights.json")
        with open(weights_path, "w") as f:
            json.dump(best_weights, f)
        mlflow.log_artifact(weights_path)

        # Create final model
        proba_lstm = Input(shape=(NUM_CLASSES,))
        proba_vgg16 = Input(shape=(NUM_CLASSES,))
        
        weighted_proba = tf.keras.layers.Lambda(
            lambda x, w1=best_weights[0], w2=best_weights[1]: w1 * x[0] + w2 * x[1]
        )([proba_lstm, proba_vgg16])

        final_model = Model(inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba)
        final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Save and log final model
        final_path = os.path.join(ARTIFACTS_DIR, "concatenate.keras")
        final_model.save(final_path)
        mlflow.log_artifact(final_path)

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts/concatenate.keras"
        mlflow.register_model(model_uri, "RegisteredConcatenateModel")

        # Save metadata
        sync_db.model_metadata.insert_one({
            "training_date": datetime.now().isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "model_version": "1.0",
            "training_metrics": {
                "accuracy": float(best_accuracy),
                "lstm_weight": float(best_weights[0]),
                "vgg16_weight": float(best_weights[1])
            },
            "data_info": {
                "train_samples": len(X_train),
                "val_samples": len(X_val)
            }
        })

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error in train_and_save_models: {e}")
        raise