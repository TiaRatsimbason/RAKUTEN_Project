# Standard library imports
import json
import os
import logging
from datetime import datetime
import io
from gridfs import GridFS

# Third-party imports
import mlflow
import mlflow.tensorflow
from mlflow.types.schema import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
import tensorflow as tf
from mlflow.tracking import MlflowClient
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
from src.scripts.data.gridfs_image_handler import GridFSImageHandler

BATCH_SIZE = 23
NUM_CLASSES = 27
ARTIFACTS_DIR = "./models"
NUMBER_OF_EPOCHS = 1 # By default

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
            
            # Save tokenizer config
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            tokenizer_path = os.path.join(ARTIFACTS_DIR, "tokenizer_config.json")
            with open(tokenizer_path, "w", encoding="utf-8") as json_file:
                json_file.write(self.tokenizer.to_json())
            
            logger.info("Logging tokenizer configuration to MLFlow")
            mlflow.log_artifact(tokenizer_path)

            # Prepare sequences
            logger.info("Preparing sequences")
            train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
            train_padded = pad_sequences(
                train_sequences, 
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post"
            )

            val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
            val_padded = pad_sequences(
                val_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post"
            )

            # Create and train model
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

            # Train
            history = self.model.fit(
                train_padded,
                tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES),
                epochs=NUMBER_OF_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(
                    val_padded,
                    tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
                ),
                callbacks=callbacks
            )

            # Log metrics
            for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                mlflow.log_metric(f"lstm_{metric}", history.history[metric][-1])

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
            input_example = train_padded[:1]
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
        self.fs = GridFS(sync_db)  # Utilisez GridFS pour accéder aux fichiers
        logger.info("ImageVGG16Model initialized")

    def create_gridfs_generator(self, df, labels, datagen, split_name):
        target_size = (224, 224)
        
        def generate_batches():
            while True:
                indices = np.random.permutation(len(df))
                for start in range(0, len(df), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(df))
                    batch_indices = indices[start:end]

                    batch_images = np.zeros((len(batch_indices), *target_size, 3))
                    batch_labels = np.zeros((len(batch_indices), NUM_CLASSES))

                    batch_df = df.iloc[batch_indices]
                    for i, (idx, row) in enumerate(batch_df.iterrows()):
                        try:
                            # Utiliser 'gridfs_file_id' pour accéder à l'image dans GridFS
                            gridfs_file_id = row['gridfs_file_id']
                            if gridfs_file_id:
                                grid_out = self.fs.get(ObjectId(gridfs_file_id))
                                img = Image.open(io.BytesIO(grid_out.read()))
                                img = img.resize(target_size)
                                img_array = img_to_array(img)
                                img_array = preprocess_input(img_array)

                                batch_images[i] = img_array
                                batch_labels[i] = tf.keras.utils.to_categorical(labels.iloc[idx], NUM_CLASSES)
                            else:
                                logger.warning(f"Image not found for row with index {idx}")

                        except Exception as e:
                            logger.error(f"Error loading image with gridfs_file_id {gridfs_file_id}: {e}")
                            continue
                    
                    yield batch_images, batch_labels
        
        return generate_batches()

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        try:
            # Log hyperparameters
            current_run = mlflow.active_run()
            if current_run and not mlflow.get_run(current_run.info.run_id).data.params:
                mlflow.log_param("batch_size", BATCH_SIZE)
                mlflow.log_param("num_classes", NUM_CLASSES)

            # Create data generators
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

            train_generator = self.create_gridfs_generator(X_train, y_train, train_datagen, "train")
            val_generator = self.create_gridfs_generator(X_val, y_val, val_datagen, "validation")

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

            steps_per_epoch = len(X_train) // BATCH_SIZE
            validation_steps = len(X_val) // BATCH_SIZE

            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=NUMBER_OF_EPOCHS,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks
            )

            # Log metrics
            for metric, values in history.history.items():
                mlflow.log_metric(f"vgg16_{metric}", values[-1])

            # Log model with signature
            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 224, 224, 3), "image_input")
            ])
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, NUM_CLASSES), "output_layer")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Get example input
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
        self.fs = GridFS(sync_db)  # Utilisez GridFS pour accéder aux images
        logger.info("Concatenate model initialized")

    def preprocess_image_from_gridfs(self, gridfs_file_id, target_size=(224, 224)):
        try:
            # Utiliser GridFS pour accéder à l'image
            grid_out = self.fs.get(ObjectId(gridfs_file_id))
            img = Image.open(io.BytesIO(grid_out.read()))
            img = img.resize(target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image with gridfs_file_id {gridfs_file_id}: {str(e)}")
            # Retourner une image noire si une erreur survient
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
            for gridfs_file_id in new_X_train["gridfs_file_id"]:
                img_array = self.preprocess_image_from_gridfs(gridfs_file_id)
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

            return best_weights, best_accuracy

        except Exception as e:
            logger.error(f"Error in weight optimization: {str(e)}")
            raise

def train_and_save_models(X_train, y_train, X_val, y_val):
    try:
        
        # Charger les données de MongoDB avec les références GridFS et 'image_path'
        train_data = pd.DataFrame(list(sync_db.labeled_train.find({}, {'_id': 0})))
        val_data = pd.DataFrame(list(sync_db.labeled_val.find({}, {'_id': 0})))

        # Séparer features et labels
        X_train = train_data.drop(['label'], axis=1)
        y_train = train_data['label']
        X_val = val_data.drop(['label'], axis=1)
        y_val = val_data['label']
        
        # Validation des entrées
        required_columns = ['description', 'gridfs_file_id']
        if not all(col in X_train.columns for col in required_columns):
            raise ValueError(f"Missing required columns in X_train: {required_columns}")
        if not all(col in X_val.columns for col in required_columns):
            raise ValueError(f"Missing required columns in X_val: {required_columns}")
            
        # Vérifier les dimensions
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Mismatch between features and labels dimensions")

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
        # Enregistrer le modèle
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts/concatenate.keras"
        model_info = mlflow.register_model(model_uri, "RegisteredConcatenateModel")

        # Attendre la création de la version (MLflow attend que le modèle soit disponible)
        client = MlflowClient()
        model_version = client.get_model_version(
            name="RegisteredConcatenateModel",
            version=model_info.version
        )

        # La version est récupérée ici
        version_number = model_version.version

        # Save metadata
        sync_db.model_metadata.insert_one({
            "training_date": datetime.now().isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "model_version": version_number,
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
