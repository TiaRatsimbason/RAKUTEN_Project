import pytest
from unittest.mock import patch, AsyncMock, MagicMock, create_autospec
from fastapi.testclient import TestClient
import pandas as pd
from src.api.app import app
import asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket
import nest_asyncio

# Permettre le nested event loop
nest_asyncio.apply()

class AsyncCursor:
    """Classe pour simuler un curseur async MongoDB"""
    def __init__(self, data):
        self.data = data
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self.data):
            raise StopAsyncIteration
        result = self.data[self._index]
        self._index += 1
        return result

@pytest.fixture
def test_data():
    """Données de test communes avec structure complète"""
    return [{
        'productid': str(i),
        'imageid': str(i),
        'description': f'test description {i}',
        'designation': f'test designation {i}',
        'prdtypecode': i % 27,
        'label': i % 27,
        'gridfs_file_id': f'file_{i}'
    } for i in range(10)]

@pytest.fixture
def mock_mongodb(test_data):
    """Mock MongoDB avec données complètes"""
    mock_db = AsyncMock(spec=AsyncIOMotorDatabase)
    
    # Configuration des collections
    collections = {
        'preprocessed_x_train': pd.DataFrame(test_data),
        'preprocessed_y_train': pd.DataFrame([{'prdtypecode': x['prdtypecode']} for x in test_data]),
        'preprocessed_x_test': pd.DataFrame(test_data),
        'labeled_test': pd.DataFrame(test_data),
        'labeled_train': pd.DataFrame(test_data),
        'labeled_val': pd.DataFrame(test_data),
        'pipeline_metadata': [],
        'data_pipeline': [],
        'model_metadata': [],
        'fs.files': test_data,
        'fs.chunks': test_data
    }

    # Mock du GridFS
    mock_fs = AsyncMock(spec=AsyncIOMotorGridFSBucket)
    mock_fs.upload_from_stream = AsyncMock(return_value='new_file_id')
    mock_db.fs = mock_fs
    
    # Mock des méthodes de collection
    async def mock_find(*args, **kwargs):
        collection_name = getattr(args[0], 'name', 'preprocessed_x_train')
        data = collections.get(collection_name, test_data)
        return AsyncCursor(data)

    async def mock_count_documents(*args, **kwargs):
        return len(test_data)

    async def mock_insert_one(*args, **kwargs):
        return AsyncMock(inserted_id='test_id')

    async def mock_insert_many(*args, **kwargs):
        return None

    async def mock_drop(*args, **kwargs):
        return None

    async def mock_create_index(*args, **kwargs):
        return 'index_name'

    async def mock_list_collection_names(*args, **kwargs):
        return list(collections.keys())

    # Configuration des collections
    for name in collections:
        collection = AsyncMock()
        collection.name = name
        collection.find = AsyncMock(side_effect=mock_find)
        collection.count_documents = AsyncMock(side_effect=mock_count_documents)
        collection.insert_one = AsyncMock(side_effect=mock_insert_one)
        collection.insert_many = AsyncMock(side_effect=mock_insert_many)
        collection.drop = AsyncMock(side_effect=mock_drop)
        collection.create_index = AsyncMock(side_effect=mock_create_index)
        setattr(mock_db, name, collection)

    mock_db.list_collection_names = AsyncMock(side_effect=mock_list_collection_names)
    return mock_db

@pytest.fixture
def client(mock_mongodb):
    """Client de test avec override des dépendances"""
    app.dependency_overrides = {}

    async def mock_process_text_chunk(chunk):
        return chunk

    with patch('src.config.mongodb.async_db', mock_mongodb), \
         patch('src.config.mongodb.sync_db', mock_mongodb), \
         patch('src.scripts.features.build_features.DataImporter'), \
         patch('src.scripts.features.build_features.TextPreprocessor'), \
         patch('src.api.routes.model.process_text_chunk', mock_process_text_chunk), \
         patch('src.api.routes.model.initialize_nltk'), \
         patch('src.scripts.main.train_and_save_model'):

        # Créer un nouveau loop event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        client = TestClient(app)
        yield client
        
        # Nettoyer le loop event
        loop.close()

@pytest.mark.asyncio
async def test_load_data(client):
    """Test de l'endpoint load-data"""
    with patch('src.scripts.data.mongodb_data_loader.MongoDBDataLoader') as mock_loader:
        mock_loader_instance = AsyncMock()
        mock_loader_instance.load_all_data.return_value = True
        mock_loader.return_value = mock_loader_instance
        
        response = client.post('/api/model/load-data/')
        assert response.status_code == 200
        assert "Data loading completed successfully" in response.json()["message"]

@pytest.mark.asyncio
async def test_data_status(client):
    """Test de l'endpoint data-status"""    
    response = client.get('/api/model/data-status/')
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["tabular_data", "images", "storage_details", "is_ready"])

@pytest.mark.asyncio
async def test_prepare_data(client, mock_mongodb):
    """Test de l'endpoint prepare-data"""
    response = client.post('/api/model/prepare-data/')
    assert response.status_code == 200
    assert response.json()["message"] == "Data preparation completed successfully"

@pytest.mark.asyncio
async def test_train_model(client):
    """Test de l'endpoint train-model"""
    with patch('src.scripts.main.train_and_save_model') as mock_train:
        mock_train.return_value = None
        response = client.post('/api/model/train-model/')
        assert response.status_code == 200
        assert response.json()["message"] == "Model training completed successfully"

@pytest.mark.asyncio
async def test_predict(client):
    """Test de l'endpoint predict"""
    with patch('src.scripts.predict.load_predictor') as mock_predictor:
        predictor = MagicMock()
        predictor.predict.return_value = {str(i): i % 27 for i in range(10)}
        mock_predictor.return_value = predictor
        
        response = client.post('/api/model/predict/?version=1')
        assert response.status_code == 200
        assert "predictions" in response.json()

@pytest.mark.asyncio
async def test_evaluate_model(client, mock_mongodb, test_data):
    """Test de l'endpoint evaluate-model"""
    with patch('src.scripts.predict.load_predictor') as mock_predictor:
        predictor = MagicMock()
        predictor.predict.return_value = {str(i): i % 27 for i in range(10)}
        mock_predictor.return_value = predictor
        
        response = client.post('/api/model/evaluate-model/?version=1')
        assert response.status_code == 200
        result = response.json()
        assert all(key in result for key in ["metrics", "mean_inference_time_ms"])

@pytest.mark.asyncio
async def test_error_handling(client, mock_mongodb):
    """Test de la gestion des erreurs"""
    # Setup
    error_msg = "Test error"
    
    # Test d'erreur pour prepare-data
    with patch('src.scripts.features.build_features.DataImporter') as mock_importer:
        mock_importer_instance = MagicMock()
        mock_importer_instance.load_data.side_effect = Exception(error_msg)
        mock_importer.return_value = mock_importer_instance
        
        response = client.post('/api/model/prepare-data/')
        assert response.status_code == 500
        assert error_msg in response.json()["detail"]

    # Test d'erreur pour predict sans version
    response = client.post('/api/model/predict/')
    assert response.status_code == 422