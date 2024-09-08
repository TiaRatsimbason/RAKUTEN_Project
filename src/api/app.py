from flask import Flask, jsonify
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from .auth import auth_blueprint  
from predict import predict_blueprint  
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from subprocess import run, CalledProcessError
import time

# Configuration du logging
logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Créer un pool de threads pour exécuter l'entraînement en parallèle
executor = ThreadPoolExecutor(2)

app = Flask(__name__)

# Configuration de JWT
app.config["JWT_SECRET_KEY"] = "your_secret_key"  # !!! A modifier au besoin !!!
jwt = JWTManager(app)

# Enregistrement des Blueprints
app.register_blueprint(auth_blueprint, url_prefix="/auth")  # Routes pour l'authentification
app.register_blueprint(predict_blueprint, url_prefix="/api")  # Routes pour les prédictions

# Variable pour suivre l'état de l'entraînement
training_in_progress = False

# Fonction pour vérifier si l'utilisateur est admin
def is_admin():
    current_user = get_jwt_identity()
    return current_user["role"] == "admin"

# Fonction pour vérifier la présence des modèles
def check_models():
    return os.path.exists("models/best_lstm_model.h5") and os.path.exists("models/best_vgg16_model.h5")

# Fonction pour installer les données et entraîner les modèles
def setup_data_and_train():
    try:
        logging.info("Lancement de l'installation des données et de l'entraînement.")
        start_time = time.time()  

        # Exécuter les scripts d'installation et d'entraînement
        run(["python", "src/data/import_raw_data.py"], check=True)
        logging.info(f"Installation des données terminée en {time.time() - start_time:.2f} secondes")

        start_time = time.time()  # Recommencer le chronomètre pour l'entraînement
        run(["python", "src/main.py"], check=True)
        logging.info(f"Entraînement des modèles terminé en {time.time() - start_time:.2f} secondes")

    except CalledProcessError as e:
        logging.error(f"Erreur pendant la configuration des données ou l'entraînement : {str(e)}")
        raise e

# Avant chaque requête, vérifier si les modèles sont présents ou lancer l'entraînement
@app.before_request
def check_and_prepare():
    if not check_models():
        logging.info("Les modèles n'existent pas. Lancement de l'installation des données et de l'entraînement.")
        setup_data_and_train()

# Fonction pour l'entraînement des modèles
def train_model():
    global training_in_progress
    training_in_progress = True
    logging.info("Début de l'entraînement du modèle LSTM et VGG...")  # Log de début d'entraînement
    
    try:
        start_time = time.time()

        # Entraînement du modèle LSTM
        logging.info("Début de l'entraînement du modèle LSTM...")
        # Appel de la fonction d'entraînement du modèle LSTM
        # lstm_model.train(X_train, y_train, X_val, y_val)
        time.sleep(10)  # Simule l'entraînement LSTM
        logging.info(f"Modèle LSTM terminé en {time.time() - start_time:.2f} secondes")

        start_time = time.time()

        # Entraînement du modèle VGG
        logging.info("Début de l'entraînement du modèle VGG...")
        # Appel de la fonction d'entraînement du modèle VGG
        # vgg_model.train(X_train, y_train, X_val, y_val)
        time.sleep(10)  # Simule l'entraînement VGG
        logging.info(f"Modèle VGG terminé en {time.time() - start_time:.2f} secondes")

    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement : {str(e)}")  # Log des erreurs
    finally:
        training_in_progress = False
        logging.info("Fin de l'entraînement.")  # Log de fin d'entraînement

# Route pour démarrer l'entraînement
@app.route('/train', methods=['POST'])
@jwt_required()  # Protection par JWT
def train():
    if not is_admin():
        return jsonify({"message": "Accès non autorisé. Administrateurs uniquement."}), 403
    
    if not training_in_progress:
        executor.submit(train_model)  # Lance l'entraînement dans un thread séparé
        return jsonify({"message": "L'entraînement est en cours."}), 202
    else:
        return jsonify({"message": "Un entraînement est déjà en cours."}), 409

# Route pour vérifier le statut de l'entraînement
@app.route('/train/status', methods=['GET'])
@jwt_required()  # JWT requis pour l'accès
def train_status():
    if training_in_progress:
        return jsonify({"status": "Entraînement en cours"}), 200
    else:
        return jsonify({"status": "Entraînement terminé"}), 200

# Route d'accueil pour vérifier que l'API fonctionne
@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur l'API de gestion de modèles et prédictions."})

# Démarrage de l'application Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
