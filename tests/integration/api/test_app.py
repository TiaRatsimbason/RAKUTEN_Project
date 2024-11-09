import pytest
import asyncio
import concurrent.futures
from unittest.mock import patch, AsyncMock, MagicMock, create_autospec, mock_open
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from src.api.app import app
from src.scripts.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_mongodb():
    mock_db = AsyncMock()
    
    # Configuration des collections
    collections = {
        'preprocessed_x_train': AsyncMock(),
        'preprocessed_y_train': AsyncMock(),
        'preprocessed_x_test': AsyncMock(),
        'predictions': AsyncMock(),
        'labeled_test': AsyncMock(),
        'labeled_train': AsyncMock(),
        'labeled_val': AsyncMock(),
        'pipeline_metadata': AsyncMock(),
        'data_pipeline': AsyncMock(),
        'model_metadata': AsyncMock()
    }
    
    # Configuration des méthodes de collection
    for collection in collections.values():
        collection.insert_one.return_value = AsyncMock(inserted_id='test_id')
        collection.find_one.return_value = {'_id': '1', 'data': 'test'}
        collection.find.return_value.to_list.return_value = [{'_id': '1', 'data': 'test'}]
        collection.create_index.return_value = 'index_name'
        collection.drop.return_value = None
    
    # Configuration de l'accès aux collections
    mock_db.__getitem__.side_effect = lambda x: collections[x]
    mock_db.get_collection.side_effect = lambda x: collections[x]
    mock_db.list_collection_names = AsyncMock(return_value=list(collections.keys()))
    
    # Configuration GridFS
    mock_db.fs = AsyncMock()
    mock_db.fs.files = MagicMock()
    mock_db.fs.files.find_one.return_value = {
        "_id": "test_id",
        "filename": "test.jpg",
        "metadata": {"original_path": "/test/"}
    }
    mock_db.fs.chunks = MagicMock()
    mock_db.fs.upload_from_stream = AsyncMock(return_value="new_file_id")
    mock_db.fs.get = AsyncMock(return_value=AsyncMock(
        read=AsyncMock(return_value=b"fake_image_data")
    ))
    
    return mock_db

def test_load_data(client, mock_mongodb):
    with patch('src.scripts.data.mongodb_data_loader.MongoDBDataLoader') as mock_loader, \
         patch('src.api.routes.model.async_db', mock_mongodb), \
         patch('src.api.routes.model.db', mock_mongodb), \
         patch('os.path.exists', return_value=True):
        mock_loader.return_value.load_all_data.return_value = True
        response = client.post('/api/model/load-data/')
        assert response.status_code == 200

def test_data_status(client, mock_mongodb):
    """Test de l'endpoint data-status"""
    with patch('src.api.routes.model.async_db', mock_mongodb), \
         patch('src.api.routes.model.db', mock_mongodb):
         
        # Configurer correctement le mock pour fs.files
        mock_mongodb.fs.files.count_documents = AsyncMock(return_value=10)
        mock_mongodb.list_collection_names.return_value = ['preprocessed_x_train', 'preprocessed_x_test', 'preprocessed_y_train']
        
        # Mock pour les collections
        for collection in ['preprocessed_x_train', 'preprocessed_y_train', 'preprocessed_x_test']:
            mock_mongodb[collection].count_documents = AsyncMock(return_value=100)
        
        response = client.get('/api/model/data-status/')
        assert response.status_code == 200

def create_sync_mongo_cursor(data):
    """Crée un curseur MongoDB sync simulé"""
    cursor = MagicMock()
    cursor.to_list = MagicMock(return_value=data)
    return cursor

def create_async_mongo_cursor(data):
    """Crée un curseur MongoDB async simulé"""
    async def async_gen():
        for item in data:
            yield item
    cursor = AsyncMock()
    cursor.__aiter__.side_effect = async_gen
    cursor.to_list = AsyncMock(return_value=data)
    return cursor

@pytest.fixture
def mock_mongodb():
    """Fixture améliorée pour mock MongoDB"""
    mock_db = AsyncMock()
    
    # Configuration des collections avec des données de test
    test_data = [{'_id': '1', 'data': 'test'} for _ in range(10)]
    
    collections = {
        'preprocessed_x_train': AsyncMock(),
        'preprocessed_y_train': AsyncMock(),
        'preprocessed_x_test': AsyncMock(),
        'predictions': AsyncMock(),
        'labeled_test': AsyncMock(),
        'labeled_train': AsyncMock(),
        'labeled_val': AsyncMock(),
        'pipeline_metadata': AsyncMock(),
        'data_pipeline': AsyncMock(),
        'model_metadata': AsyncMock(),
        'fs.files': AsyncMock(),
        'fs.chunks': AsyncMock()
    }
    
    # Configuration des méthodes pour chaque collection
    for name, collection in collections.items():
        cursor_mock = MagicMock()
        cursor_mock.__iter__.return_value = test_data
        cursor_mock.to_list.return_value = test_data
        collection.find.return_value = cursor_mock
        collection.find_one.return_value = test_data[0]
        collection.insert_one.return_value = AsyncMock(inserted_id='test_id')
        collection.insert_many.return_value = AsyncMock()
        collection.count_documents.return_value = len(test_data)
        collection.drop.return_value = None
    
    # Configuration de GridFS
    mock_db.fs = AsyncMock()
    mock_db.fs.files = collections['fs.files']
    mock_db.fs.chunks = collections['fs.chunks']
    mock_db.fs.find_one.return_value = {
        '_id': 'test_id',
        'filename': 'test.jpg',
        'metadata': {'original_path': '/image_test/'}
    }
    mock_db.fs.upload_from_stream = AsyncMock(return_value='new_file_id')
    
    # Configuration des méthodes de base de données
    mock_db.__getitem__.side_effect = lambda x: collections.get(x, AsyncMock())
    mock_db.list_collection_names = AsyncMock(return_value=list(collections.keys()))
    
    return mock_db

def test_prepare_data(client, mock_mongodb):
    """Test de l'endpoint prepare-data avec mocks améliorés"""
    with patch('src.scripts.features.build_features.DataImporter') as mock_importer, \
         patch('src.config.mongodb.MongoClient') as mock_client, \
         patch('src.config.mongodb.sync_db', mock_mongodb), \
         patch('src.config.mongodb.async_db', mock_mongodb), \
         patch('src.api.routes.model.async_db', mock_mongodb), \
         patch('src.api.routes.model.db', mock_mongodb):

        # Configuration du mock MongoClient
        mock_client.return_value = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_mongodb

        # Setup des données de test
        df = pd.DataFrame({
            'productid': [str(i) for i in range(10)],
            'imageid': [str(i) for i in range(10)],
            'description': [f'test description {i}' for i in range(10)],
            'designation': [f'test designation {i}' for i in range(10)],
            'prdtypecode': [i % 27 for i in range(10)]
        })

        # Configuration du mock DataImporter
        mock_importer_instance = MagicMock()
        mock_importer_instance.load_data.return_value = df
        mock_importer_instance.split_train_test.return_value = (
            df.copy(), df.copy(), df.copy(),
            df['prdtypecode'], df['prdtypecode'], df['prdtypecode']
        )
        mock_importer.return_value = mock_importer_instance

        response = client.post('/api/model/prepare-data/')
        assert response.status_code == 200

def test_predict(client, mock_mongodb):
    """Test de l'endpoint predict avec mocks améliorés"""
    with patch('src.scripts.predict.load_predictor') as mock_predictor, \
         patch('src.config.mongodb.MongoClient') as mock_client, \
         patch('src.config.mongodb.sync_db', mock_mongodb), \
         patch('src.config.mongodb.async_db', mock_mongodb), \
         patch('src.api.routes.model.async_db', mock_mongodb), \
         patch('src.api.routes.model.db', mock_mongodb):

        # Configuration du mock MongoClient
        mock_client.return_value = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_mongodb

        # Setup des données
        df = pd.DataFrame({
            'productid': [str(i) for i in range(10)],
            'imageid': [str(i) for i in range(10)],
            'description': [f'test description {i}' for i in range(10)],
            'designation': [f'test designation {i}' for i in range(10)]
        })

        # Configuration du mock predictor
        predictor = MagicMock()
        predictor.predict.return_value = {str(i): str(i % 27) for i in range(10)}
        mock_predictor.return_value = predictor

        # Configuration du mock de la collection
        cursor = MagicMock()
        cursor.__iter__.return_value = df.to_dict('records')
        mock_mongodb.preprocessed_x_test.find.return_value = cursor

        response = client.post('/api/model/predict/?version=1')
        assert response.status_code == 200

def test_evaluate_model(client, mock_mongodb):
    """Test de l'endpoint evaluate-model avec mocks améliorés"""
    with patch('src.scripts.features.build_features.DataImporter') as mock_importer, \
         patch('src.scripts.predict.load_predictor') as mock_predictor, \
         patch('src.config.mongodb.MongoClient') as mock_client, \
         patch('src.config.mongodb.sync_db', mock_mongodb), \
         patch('src.config.mongodb.async_db', mock_mongodb), \
         patch('src.api.routes.model.async_db', mock_mongodb), \
         patch('src.api.routes.model.db', mock_mongodb):

        # Configuration du mock MongoClient
        mock_client.return_value = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_mongodb

        # Setup des données
        df = pd.DataFrame({
            'productid': [str(i) for i in range(10)],
            'imageid': [str(i) for i in range(10)],
            'description': [f'test description {i}' for i in range(10)],
            'designation': [f'test designation {i}' for i in range(10)],
            'prdtypecode': [i % 27 for i in range(10)]
        })

        # Configuration du mock DataImporter
        mock_importer_instance = MagicMock()
        mock_importer_instance.load_data.return_value = df
        mock_importer_instance.split_train_test.return_value = (
            df.drop('prdtypecode', axis=1),
            df.drop('prdtypecode', axis=1),
            df.drop('prdtypecode', axis=1),
            df['prdtypecode'],
            df['prdtypecode'],
            df['prdtypecode']
        )
        mock_importer.return_value = mock_importer_instance

        # Configuration du mock predictor
        predictor = MagicMock()
        predictor.predict.return_value = {str(i): str(i % 27) for i in range(10)}
        mock_predictor.return_value = predictor

        # Configuration des mocks de collections
        cursor = MagicMock()
        cursor.__iter__.return_value = df.to_dict('records')
        for collection_name in ['labeled_train', 'labeled_test', 'labeled_val']:
            getattr(mock_mongodb, collection_name).find.return_value = cursor

        response = client.post('/api/model/evaluate-model/?version=1')
        assert response.status_code == 200