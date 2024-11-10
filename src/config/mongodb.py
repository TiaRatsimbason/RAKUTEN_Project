import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo import MongoClient
import gridfs

# Configuration MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:motdepasseadmin@mongo:27017/")
DB_NAME = "rakuten_db"

# Client asynchrone pour FastAPI
async_client = AsyncIOMotorClient(MONGODB_URI)
async_db = async_client[DB_NAME]

# Client synchrone pour les autres opérations
sync_client = MongoClient(MONGODB_URI)
sync_db = sync_client[DB_NAME]

# GridFS pour les images
sync_fs = gridfs.GridFS(sync_db)
async_fs = AsyncIOMotorGridFSBucket(async_db)