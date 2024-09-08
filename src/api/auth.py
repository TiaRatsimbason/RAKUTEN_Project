from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from passlib.context import CryptContext
import json
import os

auth_blueprint = Blueprint('auth', __name__)

# Gestion du hashage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Charger les utilisateurs depuis un fichier JSON
def load_users():
    users_file = os.path.join("data", "users.json")  
    if not os.path.exists(users_file):
        return []
    
    with open(users_file, "r", encoding="utf-8") as f:
        users = json.load(f)
    return users

# Sauvegarder les utilisateurs dans le fichier JSON
def save_users(users):
    users_file = os.path.join("data", "users.json")  
    os.makedirs(os.path.dirname(users_file), exist_ok=True)
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f)

# Route pour l'authentification et la génération d'un token JWT
@auth_blueprint.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    if not username or not password:
        return jsonify({"msg": "Nom d'utilisateur et mot de passe requis"}), 400

    users = load_users()

    user = next((user for user in users if user["username"] == username), None)
    if user is None or not pwd_context.verify(password, user["hashed_password"]):
        return jsonify({"msg": "Nom d'utilisateur ou mot de passe incorrect"}), 401

    access_token = create_access_token(identity={"username": username, "role": user["role"]})
    return jsonify({"access_token": access_token}), 200

# Route pour obtenir la liste des utilisateurs (seulement admin)
@auth_blueprint.route("/users", methods=["GET"])
@jwt_required()
def list_users():
    current_user = get_jwt_identity()

    if current_user["role"] != "admin":
        return jsonify({"msg": "Accès non autorisé : Administrateurs uniquement"}), 403

    users = load_users()
    return jsonify({"users": users}), 200

# Route pour enregistrer un nouvel utilisateur (seulement admin)
@auth_blueprint.route("/register", methods=["POST"])
@jwt_required()
def register_user():
    current_user = get_jwt_identity()

    if current_user["role"] != "admin":
        return jsonify({"msg": "Accès non autorisé"}), 403

    username = request.json.get("username")
    password = request.json.get("password")
    role = request.json.get("role", "user")

    if not username or not password:
        return jsonify({"msg": "Nom d'utilisateur et mot de passe requis"}), 400

    users = load_users()

    if any(user["username"] == username for user in users):
        return jsonify({"msg": "Nom d'utilisateur déjà pris"}), 400

    hashed_password = pwd_context.hash(password)
    new_user = {"username": username, "hashed_password": hashed_password, "role": role}
    users.append(new_user)

    save_users(users)
    return jsonify({"msg": "Utilisateur créé avec succès"}), 201
