import pytest
import joblib
from score import score
import requests
import subprocess
import time
import warnings
import re
import os

warnings.filterwarnings("ignore")

@pytest.fixture
def model():
    """Load the trained model fixture."""
    return joblib.load("best_model.pkl")

def test_smoke(model):
    """Check if function runs without crashing."""
    assert score("Test message", model, 0.5) is not None

def test_output_format(model):
    """Ensure function returns expected types."""
    prediction, propensity = score("Hello world", model, 0.5)
    assert type(prediction) == int
    try:
        float(propensity)
    except Exception as e:
        pytest.fail(f"score function raised an exception: {e} (Format test failed)")

def test_prediction_values(model):
    """Ensure prediction is always 0 or 1."""
    for text in ["spam", "not spam"]:
        prediction, _ = score(text, model, 0.5)
        assert prediction in (0,1)

def test_propensity_range(model):
    """Ensure propensity is between 0 and 1."""
    _, propensity = score("test", model, 0.5)
    assert 0 <= propensity <= 1

def test_threshold_behavior(model):
    """Check threshold behavior."""
    assert score("test", model, 0.0)[0] == 1  # Threshold 0 → always predict 1
    assert score("test", model, 1.0)[0] == 0  # Threshold 1 → always predict 0

def test_obvious_spam(model):
    """Ensure an obvious spam message is classified as spam."""
    assert score("Congratulations! You have won $1,000,000! Click here to claim your prize.", model, 0.5)[0] == 1

def test_obvious_non_spam(model):
    """Ensure an obvious non-spam message is classified as non-spam."""
    assert score("Hello, how are you?", model, 0.5)[0] == 0

def test_flask():

    process = subprocess.Popen(["python", "Assignment 3/app.py"], stdout=subprocess.PIPE)

    time.sleep(2)

    payload = {"text": "Hello, congratulations! You have won a prize."}
    response = requests.post("http://127.0.0.1:5001/", data=payload)

    assert response.status_code == 200
    response = requests.post("http://127.0.0.1:5001/", data={"text": "Hello, you won a prize!"})
    # Check for 'Prediction:' and 'Probability:' in the HTML response
    assert re.search(r"Prediction:\s+\w+", response.text), "Prediction not found"
    assert re.search(r"Probability:\s+\d+\.\d+", response.text), "Propensity not found"

    process.terminate()


def test_docker():
    # Step 1: Build Docker Image
    subprocess.run(["docker", "build", "-t", "flask-app", "."], check=True)

    # Step 2: Run Docker Container
    container = subprocess.Popen(
        ["docker", "run", "-p", "5001:5001", "--name", "flask-container", "flask-app"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Give the container time to start
    time.sleep(5)

    # Step 3: Send Request to Flask App
    url = "http://127.0.0.1:5001/"
    payload = {"text": "Hello, this is a test message."}
    
    response = requests.post(url, data=payload)
    
    assert response.status_code == 200    
    response = requests.post("http://127.0.0.1:5001/", data=payload)
    # Check for 'Prediction:' and 'Probability:' in the HTML response
    assert re.search(r"Prediction:\s+\w+", response.text), "Prediction not found"
    assert re.search(r"Probability:\s+\d+\.\d+", response.text), "Propensity not found"

    # Step 4: Stop and Remove the Container
    subprocess.run(["docker", "stop", "flask-container"])
    subprocess.run(["docker", "rm", "flask-container"])