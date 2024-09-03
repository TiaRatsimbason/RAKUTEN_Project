from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
import json
import os

# Configuration de la clé secrète et des paramètres JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Création du router pour l'API utilisateurs
router = APIRouter()

# Gestion du hashage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Schéma OAuth2 pour FastAPI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fonction pour détecter le chemin Google Drive
def detect_google_drive_path():
    drives = [f"{d}:" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")]
    google_drive_path = None
    for drive in drives:
        if os.path.exists(os.path.join(drive, "Mon Drive")):
            google_drive_path = os.path.join(drive, "Mon Drive")
            break
    
    if google_drive_path is None:
        raise Exception("Google Drive not found on the system.")
    
    return google_drive_path

# Définir le chemin vers la base de données utilisateurs
google_drive_path = detect_google_drive_path()
USER_DB_PATH = os.path.join(google_drive_path, "bdd-users", "users.json")

# Charger les utilisateurs depuis Google Drive
def load_users():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r") as f:
            return json.load(f)
    return {}

# Sauvegarder les utilisateurs sur Google Drive
def save_users(users):
    os.makedirs(os.path.dirname(USER_DB_PATH), exist_ok=True)
    with open(USER_DB_PATH, "w") as f:
        json.dump(users, f)

# Charger la base de données des utilisateurs en mémoire
users_db = load_users()

# Modèle pour les utilisateurs
class User(BaseModel):
    username: str
    hashed_password: str
    role: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"  # Par défaut, rôle utilisateur
    
class UserUpdate(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Fonction pour hasher un mot de passe
def get_password_hash(password):
    return pwd_context.hash(password)

# Fonction pour vérifier un mot de passe
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Fonction pour créer un token JWT
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fonction pour authentifier un utilisateur
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

# Route pour créer un utilisateur (inscription)
@router.post("/register/", response_model=User)
async def create_user(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = get_password_hash(user.password)
    user_dict = {"username": user.username, "hashed_password": hashed_password, "role": user.role}
    users_db[user.username] = user_dict
    save_users(users_db)
    return user_dict

# Route pour obtenir un token (connexion)
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# Dépendance pour obtenir l'utilisateur actuel
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Dépendance pour vérifier les rôles# Dépendance pour vérifier les rôles
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user

# Dépendance pour vérifier si l'utilisateur est un administrateur
async def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action.",
        )
    return current_user


# Route pour mettre à jour les informations d'un utilisateur
@router.put("/update/", response_model=User)
async def update_user(user_update: UserUpdate, current_user: User = Depends(get_current_active_user)):
    user = users_db.get(current_user["username"])
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_update.password:
        user["hashed_password"] = get_password_hash(user_update.password)
    if user_update.role:
        user["role"] = user_update.role
    
    users_db[current_user["username"]] = user
    save_users(users_db)
    return user


    