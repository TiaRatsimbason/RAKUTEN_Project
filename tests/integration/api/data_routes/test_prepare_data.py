import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi import status, HTTPException
from pymongo.errors import ServerSelectionTimeoutError
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

pytestmark = pytest.mark.asyncio

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

class MockAsyncCollection:
    def __init__(self, data=None):
        self.data = data or []

    async def find(self, criteria=None, projection=None):
        filtered_data = self.data
        if (criteria):
            filtered_data = [
                d for d in self.data 
                if all(d.get(k) == v for k, v in criteria.items())
            ]
        if projection and '_id' in projection:
            return [{k: v for k, v in d.items() if k != '_id'} for d in filtered_data]
        return filtered_data

    async def count_documents(self, filter_dict=None):
        if filter_dict and "metadata.split" in filter_dict:
            return len([d for d in self.data if d.get("metadata", {}).get("split") == filter_dict["metadata.split"]])
        return len(self.data)

    async def insert_one(self, document):
        self.data.append(document)
        return MagicMock(inserted_id="test_id")

    async def insert_many(self, documents):
        self.data.extend(documents)
        return MagicMock()

    async def drop(self):
        self.data = []

    async def create_index(self, keys, **kwargs):
        return "index_name"

    async def find_one(self, filter_dict):
        for doc in self.data:
            if all(doc.get(k) == v for k, v in filter_dict.items()):
                return MagicMock(
                    metadata=doc.get("metadata", {}),
                    _id="test_id",
                    read=lambda: b"test_image_data",
                    filename="test.jpg"
                )
        return None

class MockCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []

    async def insert_one(self, document):
        self.documents.append(document)

    async def drop(self):
        self.documents = []

    async def count_documents(self, query):
        # Implémentez la logique de comptage selon vos besoins
        return len(self.documents)

    async def create_index(self, index):
        pass  # Vous pouvez ajouter une logique si nécessaire

    async def find(self, *args, **kwargs):
        return self.documents

class MockGridFS:
    def __init__(self):
        self.files = MockAsyncCollection([{
            'metadata': {
                'split': 'train',
                'productid': str(i),
                'imageid': str(i)
            },
            '_id': f'test_id_{i}',
            'data': b'test_image_data'
        } for i in range(10)])
        self.chunks = MockAsyncCollection([{'data': b'test_chunk_data'}])

    async def upload_from_stream(self, filename, source, metadata=None):
        file_id = f"test_file_id_{len(self.files.data)}"
        self.files.data.append({
            '_id': file_id,
            'filename': filename,
            'metadata': metadata,
            'data': source
        })
        return file_id

    async def get(self, file_id):
        return MagicMock(
            read=lambda: b'test_image_data',
            metadata={'productid': '1', 'imageid': '1', 'split': 'train'}
        )

class MockDB:
    def __init__(self):
        self.collections = {}
        self.fs = MockGridFS()  # Ajoutez cette ligne pour inclure l'attribut fs

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]

class MockDataImporter:
    def __init__(self):
        self.db = MockDB()
        self.fs = self.db.fs

    async def load_data(self):
        # Simulate ServerSelectionTimeoutError for MongoDB connection failure
        raise ServerSelectionTimeoutError("MongoDB not reachable")

    def split_train_test(self, df, samples_per_class=600):
        train = df.iloc[:6]
        val = df.iloc[6:8]
        test = df.iloc[8:]
        y_train = train.pop('prdtypecode') if 'prdtypecode' in train else pd.Series()
        y_val = val.pop('prdtypecode') if 'prdtypecode' in val else pd.Series()
        y_test = test.pop('prdtypecode') if 'prdtypecode' in test else pd.Series()
        return train, val, test, y_train, y_val, y_test

class TestPrepareDataRoute:
    async def test_prepare_data_success(self, async_client):
        """Test le cas où la préparation des données réussit"""
        mock_db = MockDB()
        mock_data_importer = MockDataImporter()

        async def mock_prepare_data():
            return {"message": "Data preparation completed successfully"}

        with patch('src.api.routes.model.async_db', mock_db), \
             patch('src.api.routes.model.sync_db', mock_db), \
             patch('src.api.routes.model.async_fs', mock_db.fs), \
             patch('src.scripts.features.build_features.DataImporter', return_value=mock_data_importer), \
             patch('src.api.routes.model.prepare_data', new=mock_prepare_data):

            response = await async_client.post('/api/model/prepare-data/')
            assert response.status_code == status.HTTP_200_OK
            assert "Data preparation completed successfully" in response.json()["message"]

    async def test_prepare_data_mongodb_connection_error(self, async_client):
        with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_motor_client:
            # Configurez le mock pour lever une exception lors de la connexion
            mock_client_instance = mock_motor_client.return_value
            mock_client_instance.server_info.side_effect = ServerSelectionTimeoutError("MongoDB not reachable")

            response = await async_client.post('/api/model/prepare-data/')
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "MongoDB not reachable" in response.json()["detail"]

    async def test_prepare_data_error_no_images(self, async_client):
        """Test le cas où il n'y a pas d'images à traiter"""
        mock_db = MockDB()
        mock_db.fs.files = MockAsyncCollection()

        async def mock_prepare_data():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No images were found"
            )

        with patch('src.api.routes.model.async_db', mock_db), \
             patch('src.api.routes.model.sync_db', mock_db), \
             patch('src.api.routes.model.async_fs', mock_db.fs), \
             patch('src.scripts.features.build_features.DataImporter', return_value=MockDataImporter()), \
             patch('src.api.routes.model.prepare_data', new=mock_prepare_data):

            response = await async_client.post('/api/model/prepare-data/')
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "No images were found" in response.json()["detail"]

    async def test_prepare_data_error_image_processing(self, async_client):
        """Test le cas où le traitement des images échoue"""
        mock_db = MockDB()

        class ErrorGridFS(MockGridFS):
            async def upload_from_stream(self, *args, **kwargs):
                raise Exception("Image processing error")

        mock_db.fs = ErrorGridFS()

        with patch('src.api.routes.model.async_db', mock_db), \
             patch('src.api.routes.model.sync_db', mock_db), \
             patch('src.api.routes.model.async_fs', mock_db.fs), \
             patch('src.scripts.features.build_features.DataImporter', return_value=MockDataImporter()), \
             patch('src.api.routes.model.initialize_nltk'):

            response = await async_client.post('/api/model/prepare-data/')
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Image processing error" in response.json()["detail"]
