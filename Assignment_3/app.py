from flask import Flask, request, jsonify, render_template_string
import joblib
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure correct path

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        prediction, probability = score(text, model, 0.55)

        bg_color = "#dc3545" if prediction == 1 else "#28a745"

        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Spam Classifier</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        flex-direction: column;
                    }
                    .container {
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        width: 90%;
                        max-width: 400px;
                    }
                    h1 {
                        color: #333;
                        margin-bottom: 10px;
                    }
                    label {
                        font-size: 16px;
                        color: #555;
                    }
                    input[type="text"] {
                        width: 90%;
                        padding: 10px;
                        margin-top: 10px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        font-size: 16px;
                    }
                    input[type="submit"] {
                        background-color: #007BFF;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        margin-top: 10px;
                        border-radius: 5px;
                        font-size: 16px;
                        cursor: pointer;
                        transition: 0.3s;
                    }
                    input[type="submit"]:hover {
                        background-color: #0056b3;
                    }
                    .result {
                        margin-top: 20px;
                        padding: 15px;
                        border-radius: 5px;
                        font-size: 18px;
                        font-weight: bold;
                        color: white;
                        background-color: {{ bg_color }};
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Spam Classifier</h1>
                    <form action="/" method="post">
                        <label for="text">Enter Text:</label><br>
                        <input type="text" id="text" name="text" required><br><br>
                        <input type="submit" value="Check Spam">
                    </form>
                    <div class="result">
                        Text: {{ text }} <br>
                        Prediction: {{ prediction }} <br>
                        Probability: {{ probability }}
                    </div>
                </div>
            </body>
            </html>
        """, text=text, prediction=prediction, probability=round(probability, 2), bg_color=bg_color)
    
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spam Classifier</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    flex-direction: column;
                }
                .container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    width: 90%;
                    max-width: 400px;
                }
                h1 {
                    color: #333;
                    margin-bottom: 10px;
                }
                label {
                    font-size: 16px;
                    color: #555;
                }
                input[type="text"] {
                    width: 90%;
                    padding: 10px;
                    margin-top: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    font-size: 16px;
                }
                input[type="submit"] {
                    background-color: #007BFF;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    margin-top: 10px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Spam Classifier</h1>
                <form action="/" method="post">
                    <label for="text">Enter Text:</label><br>
                    <input type="text" id="text" name="text" required><br><br>
                    <input type="submit" value="Check Spam">
                </form>
            </div>
        </body>
        </html>
    """

if __name__ == "__main__":
    app.run(port=5001)

#python -m pytest --cov=score --cov=app --cov=test_  --cov-report=term test_.py > coverage.txt
