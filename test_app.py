import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "Welcome to the Home Page"

def test_predict_endpoint(client):
    """Test the prediction endpoint"""
    response = client.post('/predict', json={'feature1': 0.5, 'feature2': 1.2, 'feature3': 0.8})
    assert response.status_code == 200
    assert 'prediction' in response.json
