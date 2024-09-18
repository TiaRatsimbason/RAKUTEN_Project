# Standard library imports
import os

# Third-party library imports
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Local/application-specific imports
from routes import model

app = FastAPI()


# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)
db = client["rakuten-database"]

# Include routes
app.include_router(router=model.router, prefix="/api", tags=["model"])

"""
# Pydantic model for our data
class Item(BaseModel):
    name: str
    description: str


@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    new_item = await db.items.insert_one(item.dict())
    created_item = await db.items.find_one({"_id": new_item.inserted_id})
    return created_item


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    item = await db.items.find_one({"_id": item_id})
    if item:
        return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.get("/items/")
async def read_all_items():
    items = await db.items.find().to_list(length=100)
    return items


@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    update_result = await db.items.update_one({"_id": item_id}, {"$set": item.dict()})
    if update_result.modified_count == 1:
        updated_item = await db.items.find_one({"_id": item_id})
        return updated_item
    raise HTTPException(status_code=404, detail="Item not found")


@app.delete("/items/{item_id}")
async def delete_item(item_id: str):
    delete_result = await db.items.delete_one({"_id": item_id})
    if delete_result.deleted_count == 1:
        return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")
"""
