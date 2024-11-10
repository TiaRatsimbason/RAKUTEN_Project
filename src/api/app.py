from fastapi import FastAPI
from src.config.mongodb import async_db
from src.api.routes import model
from prometheus_fastapi_instrumentator import Instrumentator
# Local/application-specific imports

app = FastAPI()
API_URL = "/api"

Instrumentator().instrument(app).expose(app)

# Include routes
app.include_router(
    router=model.router, 
    prefix=f"{API_URL}/model", 
    tags=["model"]
)

