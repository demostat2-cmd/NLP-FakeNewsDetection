from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent

# The uploaded model/vectorizer are expected to sit next to app.py.
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Text Classifier</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 760px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.5;
        }
        textarea {
            width: 100%;
            min-height: 180px;
            padding: 12px;
            box-sizing: border-box;
            border: 1px solid #bbb;
            border-radius: 8px;
            font: inherit;
            resize: vertical;
        }
        button {
            margin-top: 12px;
            padding: 10px 18px;
            border: 0;
            border-radius: 8px;
            cursor: pointer;
            font: inherit;
        }
        .result {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            background: #f3f3f3;
        }
        .error {
            color: #a00;
        }
        .confidence {
            font-size: 1.1rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>Text Classifier</h1>
    <p>Enter text to classify it using the supplied TF-IDF vectorizer and Multinomial Naive Bayes model.</p>

    <form method="post">
        <textarea name="text" placeholder="Enter text here..." required>{{ text }}</textarea>
        <br>
        <button type="submit">Classify</button>
    </form>

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if prediction %}
        <div class="result">
            <div>Prediction: <strong>{{ prediction }}</strong></div>
            {% if confidence is not none %}
                <div class="confidence">Confidence: {{ "%.2f"|format(confidence * 100) }}%</div>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
"""


def predict_text(text: str):
    """Return the predicted class and class probabilities for one text."""
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, model.predict_proba(features)[0])
        }

    confidence = max(probabilities.values()) if probabilities else None
    return str(prediction), probabilities, confidence


@app.get("/")
def home():
    return render_template_string(
        HTML,
        text="",
        prediction=None,
        confidence=None,
        error=None,
    )


@app.post("/")
def classify_form():
    text = request.form.get("text", "").strip()

    if not text:
        return render_template_string(
            HTML,
            text="",
            prediction=None,
            confidence=None,
            error="Please enter some text.",
        ), 400

    prediction, _, confidence = predict_text(text)

    return render_template_string(
        HTML,
        text=text,
        prediction=prediction,
        confidence=confidence,
        error=None,
    )


@app.post("/predict")
def predict_api():
    """
    JSON API.

    Request:
        {"text": "some text"}

    Response:
        {
            "prediction": "Real",
            "probabilities": {"Fake": 0.31, "Real": 0.69}
        }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "JSON body must contain a non-empty 'text' string."}), 400

    prediction, probabilities, confidence = predict_text(text.strip())

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Set HOST/PORT if desired, e.g. HOST=0.0.0.0 PORT=5000.
    import os

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
