**Commandes pour lancer l'API**

## Commandes Bash (il faut un terminal avec commandes bash)##

## Pour executer l'API:##

Si l'environnement n'a pas été créé, cette commande lancera la création de celui-ci, puis l'installation des dépendances contenues dans le fichier requirements.txt dans cet environnement. Elle vérifiera également si le répertoire Data, contenant les dossiers Raw et Preprocessed, existe. Sinon, elle procédera à leur création ainsi qu'au téléchargement des données. Ensuite, elle entraînera le modèle, mais récupérera les meilleurs modèles depuis le drive. Enfin, elle lancera l'API.

"""
./src/api/automate_setup.sh

"""

## Pour charger les données en local (si elles ont été mises à jour) :##

"""
curl -X POST "http://localhost:8000/setup-data/"

"""

## Pour enregistrer un utilisateur:##

"""
headers="Content-Type: application/json"
body='{
"username": "Tia",
"password": "Tia@7777",
"role": "user"
}'
response=$(curl -X POST "http://127.0.0.1:8000/users/register/" \
    -H "$headers" \
 -d "$body")
echo "Response: $response"

"""

## Pour avoir l'access_token:

"""

# Définir les en-têtes

headers="Content-Type: application/x-www-form-urlencoded"

# Définir le corps de la requête

body="username=Tia&password=Tia@7777"

# Faire une requête HTTP POST avec curl

response=$(curl -X POST "http://localhost:8000/users/token" \
    -H "$headers" \
 --data "$body")

# Extraire le token de la réponse

token=$(echo $response | jq -r '.access_token')

# Afficher le token (si `jq` est installé pour traiter les JSON)

echo "Token: $token"

"""

## Pour entraîner le model (rôle = admin nécéssaire):

"""
curl -X POST "http://localhost:8000/train-model/" -H "Authorization: Bearer $token"

"""

## Pour faire une requête à l'api:##

"""
headers="Authorization: Bearer $token"
file_path="C:\Users\Tia\Documents\projet RAKUTEN MLOps\juin24cmlops_rakuten_2\data\preprocessed/X_test_update.csv"
##remplacer par le bon chemin sur votre PC
images_folder="C:\Users\Tia\Documents\projet RAKUTEN MLOps\juin24cmlops_rakuten_2\data\preprocessed/image_test"
##remplacer par le bon chemin sur votre PC
response=$(curl -X POST "http://localhost:8000/predict/" \
 -H "Content-Type: multipart/form-data" \
 -H "$headers" \
    -F "file=@$file_path" \
 -F "images_folder=$images_folder")
echo "Response: $response"
"""

## Pour mettre à jour un utilisateur:##

"""
headers="Authorization: Bearer $token"
body='{"role": "admin"}'
response=$(curl -X PUT "http://127.0.0.1:8000/users/update/" \
 -H "$headers" \
    -H "Content-Type: application/json" \
    -d "$body")
echo "Response: $response"

"""
