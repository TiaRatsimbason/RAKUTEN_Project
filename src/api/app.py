from fastapi import FastAPI
from src.config.mongodb import async_db
from src.api.routes import model

app = FastAPI()
API_URL = "/api"

# Include routes
app.include_router(
    router=model.router, 
    prefix=f"{API_URL}/model", 
    tags=["model"]
)