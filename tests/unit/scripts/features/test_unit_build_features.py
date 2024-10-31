import nltk
import numpy as np
import pandas as pd
import pytest

from src.scripts.features.build_features import DataImporter, ImagePreprocessor, TextPreprocessor


@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Fixture pour s'assurer que toutes les ressources NLTK nécessaires sont téléchargées"""
    try:
        # Téléchargement des ressources NLTK nécessaires
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        pytest.skip(f"Impossible de télécharger les ressources NLTK: {str(e)}")


@pytest.fixture
def data_importer():
    return DataImporter()


@pytest.fixture
def sample_data():
    """Fixture pour créer des données de test"""
    X_train_data = pd.DataFrame({
        'designation': ['Product1', 'Product2', 'Product3'],
        'description': ['Desc1', 'Desc2', 'Desc3'],
        'Unnamed: 0': [1, 2, 3],
        'productid': [101, 102, 103],
        'imageid': [201, 202, 203]
    })

    Y_train_data = pd.DataFrame({
        'Unnamed: 0': [1, 2, 3],
        'prdtypecode': ['A', 'B', 'A']
    })

    return X_train_data, Y_train_data


@pytest.fixture
def large_sample_data():
    """Fixture pour créer un grand jeu de données de test"""
    np.random.seed(42)
    n_samples = 2000
    data = pd.DataFrame({
        'description': ['desc'] * n_samples,
        'productid': range(n_samples),
        'imageid': range(n_samples),
        'prdtypecode': np.random.choice([0, 1, 2], size=n_samples)
    })
    return data


class TestDataImporter:
    def test_split_train_test(self, data_importer, large_sample_data):
        """Test de la fonction split_train_test avec vérification des dimensions"""
        X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(
            large_sample_data, samples_per_class=600
        )

        assert len(y_train) == 1800  # 600 samples * 3 classes
        assert len(y_val) == 150  # 50 samples * 3 classes

        expected_columns = {'description', 'productid', 'imageid'}
        assert expected_columns.issubset(set(X_train.columns))
        assert expected_columns.issubset(set(X_val.columns))
        assert expected_columns.issubset(set(X_test.columns))


class TestImagePreprocessor:
    @pytest.fixture
    def image_preprocessor(self):
        return ImagePreprocessor()

    @pytest.fixture
    def image_test_df(self):
        return pd.DataFrame({
            'imageid': [1, 2, 3],
            'productid': [101, 102, 103]
        })

    def test_image_path_creation(self, image_preprocessor, image_test_df):
        """Test de la création des chemins d'images"""
        image_preprocessor.preprocess_images_in_df(image_test_df)

        expected_path = 'data/preprocessed/image_train/image_1_product_101.jpg'
        assert image_test_df['image_path'].iloc[0] == expected_path
        assert image_test_df['image_path'].notna().all()


class TestTextPreprocessor:
    @pytest.fixture
    def text_preprocessor(self):
        return TextPreprocessor()

    @pytest.mark.parametrize("input_text,expected_words", [
        ("simple text", ["simple", "text"]),
        ("UPPER CASE", ["upper", "case"]),
        ("<p>HTML</p>", ["html"]),
        ("", []),
    ])
    def test_basic_text_preprocessing(self, text_preprocessor, input_text, expected_words):
        """Test du prétraitement de texte basique"""
        try:
            processed = text_preprocessor.preprocess_text(input_text)
            processed_words = processed.split()
            assert all(word in processed_words for word in expected_words)
        except Exception as e:
            pytest.fail(f"Le prétraitement a échoué avec l'erreur: {str(e)}")

    def test_nan_handling(self, text_preprocessor):
        """Test de la gestion des valeurs NaN"""
        result = text_preprocessor.preprocess_text(float('nan'))
        assert result == ""

    def test_dataframe_preprocessing(self, text_preprocessor):
        """Test du prétraitement sur un DataFrame"""
        df = pd.DataFrame({
            'text': ['Simple text', 'UPPER CASE', float('nan')],
            'other': [1, 2, 3]
        })

        try:
            text_preprocessor.preprocess_text_in_df(df, ['text'])
            assert df['text'].notna().all()
            assert all(isinstance(text, str) for text in df['text'])
        except Exception as e:
            pytest.fail(f"Le prétraitement du DataFrame a échoué avec l'erreur: {str(e)}")


def test_full_pipeline(data_importer, large_sample_data):
    """Test d'intégration du pipeline complet"""
    try:
        # Test du pipeline complet
        X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(
            large_sample_data, samples_per_class=600
        )

        # Vérification des dimensions
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Vérification des colonnes
        assert all(col in X_train.columns for col in ['description', 'productid', 'imageid'])

    except Exception as e:
        pytest.fail(f"Le pipeline a échoué avec l'erreur: {str(e)}")


if __name__ == '__main__':
    pytest.main([__file__])
