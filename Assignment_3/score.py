import joblib
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load trained model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    """ Apply the same preprocessing used during training. """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return ' '.join(tokens)

def score(text: str, model, threshold: float):
    """ Predict whether the text is spam or not based on a given threshold. """
    assert isinstance(text, str), "Input text must be a string"
    assert isinstance(threshold, (float, int)), "Threshold must be a number"

    # Preprocess text before vectorizing
    text_clean = preprocess_text(text)
    
    # Transform text using the saved vectorizer
    text_vectorized = vectorizer.transform([text_clean])  

    # Get probability and prediction
    propensity = model.predict_proba(text_vectorized)[:, 1][0]
    prediction = int(propensity >= threshold)  

    assert 0 <= propensity <= 1, "Propensity must be between 0 and 1"
    return prediction, propensity