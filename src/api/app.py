from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI()


# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)
db = client["rakuten-database"]


# Pydantic model for our data
class Item(BaseModel):
    name: str
    description: str


# Connect to MongoDB on startup
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    app.mongodb = app.mongodb_client.testdb


# Close MongoDB connection on shutdown
@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


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
