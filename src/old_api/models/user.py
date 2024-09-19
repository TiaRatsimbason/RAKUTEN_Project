from pydantic import BaseModel


class User(BaseModel):
    username: str
    hashed_password: str
    role: str = "user"  # Par défaut, rôle utilisateur
