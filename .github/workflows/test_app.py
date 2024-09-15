# test.py
import pytest
from app import app  # Import your Flask app from app.py

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data  # Adjust based on your actual home page content

def test_predict_endpoint(client):
    """Test the prediction endpoint"""
    response = client.post('/predict', json={'feature1': 10, 'feature2': 20})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    # Adjust based on your expected response
