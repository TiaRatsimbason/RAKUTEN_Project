# db_connection.py
from pymongo import MongoClient
import os

# Fonction pour obtenir la base de données
def get_database():
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongodb_uri)
    return client.get_default_database()

# Nouvelle fonction pour créer un utilisateur administrateur dans MongoDB
def create_admin_user():
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongodb_uri)

    # Connexion à la base de données d'authentification par défaut (admin)
    admin_db = client["admin"]

    # Créer l'utilisateur admin avec les rôles nécessaires
    try:
        admin_db.command("createUser", "admin",
                         pwd="motdepasseadmin",
                         roles=[
                             {"role": "readWrite", "db": "rakuten_db"},
                             {"role": "dbAdmin", "db": "rakuten_db"}
                         ])
        print("Utilisateur admin créé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création de l'utilisateur : {e}")

# Exécuter le script pour créer l'utilisateur
if __name__ == "__main__":
    create_admin_user()
