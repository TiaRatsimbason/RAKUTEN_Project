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

# Configuration
BATCH_SIZE = 23
NUM_CLASSES = 27
ARTIFACTS_DIR = "./models"

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
        logger.info("TextLSTMModel initialized")

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        try:
            # Log hyperparameters
            current_run = mlflow.active_run()
            if current_run and not mlflow.get_run(current_run.info.run_id).data.params:
                mlflow.log_param("max_words", self.max_words)
                mlflow.log_param("max_sequence_length", self.max_sequence_length)
                mlflow.log_param("batch_size", BATCH_SIZE)

            # Tokenizer configuration
            logger.info("Fitting tokenizer on training data")
            self.tokenizer.fit_on_texts(X_train["description"])
            
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            tokenizer_path = os.path.join(ARTIFACTS_DIR, "tokenizer_config.json")
            with open(tokenizer_path, "w", encoding="utf-8") as json_file:
                json_file.write(self.tokenizer.to_json())
            
            logger.info("Logging tokenizer configuration to MLFlow")
            mlflow.log_artifact(tokenizer_path)

            # Prepare sequences
            logger.info("Preparing sequences")
            train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
            train_padded_sequences = pad_sequences(
                train_sequences, 
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post"
            )

            val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
            val_padded_sequences = pad_sequences(
                val_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post"
            )

            # LSTM Model
            logger.info("Building LSTM model")
            text_input = Input(shape=(self.max_sequence_length,), name="input_layer")
            embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(text_input)
            lstm_layer = LSTM(128)(embedding_layer)
            output = Dense(NUM_CLASSES, activation="softmax", name="output_layer")(lstm_layer)

            self.model = Model(inputs=[text_input], outputs=output)
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Training
            logger.info("Training LSTM model")
            lstm_checkpoint_path = os.path.join(ARTIFACTS_DIR, "best_lstm_model.keras")
            callbacks = [
                ModelCheckpoint(filepath=lstm_checkpoint_path, save_best_only=True),
                EarlyStopping(patience=3, restore_best_weights=True),
                TensorBoard(log_dir="logs")
            ]

            history = self.model.fit(
                train_padded_sequences,
                tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES),
                epochs=1,
                batch_size=BATCH_SIZE,
                validation_data=(
                    val_padded_sequences,
                    tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
                ),
                callbacks=callbacks
            )

            # Log metrics
            for metric, values in history.history.items():
                mlflow.log_metric(f"lstm_{metric}", values[-1])

            # Define model signature
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, self.max_sequence_length), "input_layer")
            ])
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, NUM_CLASSES), "output_layer")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log model
            logger.info("Logging LSTM model to MLFlow")
            input_example = train_padded_sequences[:1]
            mlflow.tensorflow.log_model(
                self.model,
                "TextLSTMModel",
                signature=signature,
                input_example=input_example
            )
            mlflow.log_artifact("logs")

        except Exception as e:
            logger.error(f"Error in TextLSTMModel training: {str(e)}")
            raise

class ImageVGG16Model:
    def __init__(self):
        self.model = None
        logger.info("ImageVGG16Model initialized")

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        try:
            # Log hyperparameters
            current_run = mlflow.active_run()
            if current_run and not mlflow.get_run(current_run.info.run_id).data.params:
                mlflow.log_param("batch_size", BATCH_SIZE)
                mlflow.log_param("num_classes", NUM_CLASSES)

            # Prepare data generators
            logger.info("Setting up data generators")
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                validation_split=0.2
            )

            # Vérifier les chemins d'images
            logger.info("Checking image paths...")
            valid_train_data = X_train[X_train['image_path'].apply(os.path.exists)].copy()
            valid_val_data = X_val[X_val['image_path'].apply(os.path.exists)].copy()

            if len(valid_train_data) == 0 or len(valid_val_data) == 0:
                raise ValueError("No valid image paths found")

            logger.info(f"Valid training images: {len(valid_train_data)}")
            logger.info(f"Valid validation images: {len(valid_val_data)}")

            # Convertir les labels en strings pour le générateur
            valid_train_data['prdtypecode'] = y_train[valid_train_data.index].astype(str)
            valid_val_data['prdtypecode'] = y_val[valid_val_data.index].astype(str)

            train_generator = train_datagen.flow_from_dataframe(
                dataframe=valid_train_data,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                shuffle=True
            )

            val_generator = train_datagen.flow_from_dataframe(
                dataframe=valid_val_data,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                shuffle=False
            )

            # Build VGG16 model
            logger.info("Building VGG16 model")
            image_input = Input(shape=(224, 224, 3), name="image_input")
            base_model = VGG16(
                include_top=False,
                weights="imagenet",
                input_tensor=image_input
            )

            # Freeze VGG16 layers
            for layer in base_model.layers:
                layer.trainable = False

            x = base_model.output
            x = Flatten()(x)
            x = Dense(256, activation="relu")(x)
            output = Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

            self.model = Model(inputs=base_model.input, outputs=output)
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Training
            logger.info("Training VGG16 model")
            vgg_checkpoint_path = os.path.join(ARTIFACTS_DIR, "best_vgg16_model.keras")
            callbacks = [
                ModelCheckpoint(filepath=vgg_checkpoint_path, save_best_only=True),
                EarlyStopping(patience=3, restore_best_weights=True),
                TensorBoard(log_dir="logs")
            ]

            history = self.model.fit(
                train_generator,
                epochs=1,
                validation_data=val_generator,
                callbacks=callbacks
            )

            # Log metrics
            for metric, values in history.history.items():
                mlflow.log_metric(f"vgg16_{metric}", values[-1])

            # Define model signature
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 224, 224, 3), "image_input")
            ])
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, NUM_CLASSES), "output_layer")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log model
            logger.info("Logging VGG16 model to MLFlow")
            # Prendre un batch d'exemple
            input_example = next(train_generator)[0][:1]
            
            mlflow.tensorflow.log_model(
                self.model,
                "ImageVGG16Model",
                signature=signature,
                input_example=input_example
            )
            mlflow.log_artifact("logs")

        except Exception as e:
            logger.error(f"Error in ImageVGG16Model training: {str(e)}")
            raise

class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        logger.info("Concatenate model initialized")

    def preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                # Retourner une image noire si le fichier n'existe pas
                return np.zeros((*target_size, 3))
            
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return np.zeros((*target_size, 3))

    def predict(self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10):
        try:
            logger.info("Starting prediction for concatenate model")
            num_classes = NUM_CLASSES

            # Resampling
            new_X_train = pd.DataFrame(columns=X_train.columns)
            new_y_train = pd.Series(dtype=int)

            # Resample data for each class
            for class_label in range(num_classes):
                indices = np.where(y_train == class_label)[0]
                if len(indices) > 0:
                    n_samples = min(new_samples_per_class, len(indices))
                    sampled_indices = resample(indices, n_samples=n_samples, replace=False, random_state=42)
                    new_X_train = pd.concat([new_X_train, X_train.iloc[sampled_indices]])
                    new_y_train = pd.concat([new_y_train, y_train.iloc[sampled_indices]])

            new_X_train = new_X_train.reset_index(drop=True)
            new_y_train = new_y_train.reset_index(drop=True)

            logger.info(f"Resampled dataset size: {len(new_X_train)}")

            # Text preprocessing
            logger.info("Processing text data")
            train_sequences = self.tokenizer.texts_to_sequences(new_X_train["description"])
            train_padded_sequences = pad_sequences(
                train_sequences,
                maxlen=max_sequence_length,
                padding="post",
                truncating="post"
            )

            # Image preprocessing
            logger.info("Processing image data")
            images_train = []
            for image_path in new_X_train["image_path"]:
                img_array = self.preprocess_image(image_path)
                images_train.append(img_array)
            
            images_train = np.array(images_train)
            
            if len(images_train) == 0:
                raise ValueError("No valid images found for training")

            # Make predictions
            logger.info("Making predictions with individual models")
            lstm_proba = self.lstm.predict(train_padded_sequences, verbose=1)
            vgg16_proba = self.vgg16.predict(images_train, verbose=1)

            return lstm_proba, vgg16_proba, new_y_train.values

        except Exception as e:
            logger.error(f"Error in concatenate predict: {str(e)}")
            raise

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        try:
            logger.info("Starting weight optimization")
            best_weights = None
            best_accuracy = 0.0

            # Chercher les meilleurs poids
            for lstm_weight in np.linspace(0, 1, 101):
                vgg16_weight = 1.0 - lstm_weight
                combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
                final_predictions = np.argmax(combined_predictions, axis=1)
                accuracy = accuracy_score(y_train, final_predictions)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = (lstm_weight, vgg16_weight)

            logger.info(f"Best weights found: LSTM={best_weights[0]:.3f}, VGG16={best_weights[1]:.3f}")
            logger.info(f"Best accuracy: {best_accuracy:.3f}")

            # Log dans MLflow
            mlflow.log_param("best_lstm_weight", best_weights[0])
            mlflow.log_param("best_vgg16_weight", best_weights[1])
            mlflow.log_metric("best_combined_accuracy", best_accuracy)

            return best_weights

        except Exception as e:
            logger.error(f"Error in weight optimization: {str(e)}")
            raise