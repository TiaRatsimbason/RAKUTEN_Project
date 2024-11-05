# Standard library imports
import os

# Third-party library imports
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from src.api.routes import model
from prometheus_fastapi_instrumentator import Instrumentator
# Local/application-specific imports


app = FastAPI()

API_URL = "/api"

Instrumentator().instrument(app).expose(app)
# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)
db = client["rakuten-database"]

# Include routes
app.include_router(router=model.router, prefix=f"{API_URL}/model", tags=["model"])

