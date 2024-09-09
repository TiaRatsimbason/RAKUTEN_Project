# db_connection.py
from pymongo import MongoClient
import os


def get_database():
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongodb_uri)
    return client.get_default_database()
