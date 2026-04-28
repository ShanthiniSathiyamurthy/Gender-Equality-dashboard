from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import os
import time
import subprocess
from datetime import datetime

app = Flask(__name__)

# =========================================
# 1. PATHS
# =========================================
MODEL_PATH = "promotion_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "model_features.pkl"
ENCODERS_PATH = "label_encoders.pkl"
LOG_FILE = "prediction_logs.csv"
DATA_PATH = "cleaned_gender_data.csv"
PIPELINE_SCRIPT = "train_model.py"

# =========================================
# 2. LOAD ARTIFACTS
# =========================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

print("✅ Model loaded successfully")


# =========================================
# 3. INIT LOG FILE
# =========================================
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp","prediction","probability"]).to_csv(LOG_FILE, index=False)


# =========================================
# 4. PREPROCESS FUNCTION
# =========================================
def preprocess(data):

    df = pd.DataFrame([data])

    cat_cols = ['Gender','Department','Job_Level','Education','Performance_Rating']

    for col in cat_cols:
        if col in df.columns:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                df[col] = 0

    df = df.reindex(columns=features, fill_value=0)

    return df


# =========================================
# 5. RETRAIN FUNCTION
# =========================================
def retrain_model():

    print("🚀 Retraining started...")

    try:
        process = subprocess.run(
            ["python", PIPELINE_SCRIPT],
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            print("✅ Retraining successful")

            # reload model
            global model, scaler, features, label_encoders

            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            features = joblib.load(FEATURES_PATH)
            label_encoders = joblib.load(ENCODERS_PATH)

            return True

        else:
            print("❌ Retrain error:", process.stderr)
            return False

    except Exception as e:
        print("Exception:", e)
        return False


# =========================================
# 6. HOME (API INFO)
# =========================================
@app.route('/')
def home():
    return {
        "status": "running",
        "project": "Gender Equality MLOps System",
        "endpoints": {
            "predict": "/predict",
            "monitor": "/monitor",
            "dashboard": "/dashboard"
        }
    }


# =========================================
# 7. PREDICTION API
# =========================================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.json

        df = preprocess(data)
        scaled = scaler.transform(df)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        # LOGGING
        log = pd.DataFrame([{
            "timestamp": datetime.now(),
            "prediction": int(pred),
            "probability": float(prob)
        }])

        log.to_csv(LOG_FILE, mode='a', header=False, index=False)

        # AUTO RETRAIN CONDITION
        logs = pd.read_csv(LOG_FILE)

        retrain_trigger = False

        if len(logs) % 500 == 0 and len(logs) > 0:
            retrain_trigger = retrain_model()

        return jsonify({
            "prediction": "Promoted" if pred == 1 else "Not Promoted",
            "probability": round(float(prob), 4),
            "retrain_triggered": retrain_trigger,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================================
# 8. MONITOR API
# =========================================
@app.route('/monitor')
def monitor():

    if not os.path.exists(LOG_FILE):
        return {"message": "No logs yet"}

    logs = pd.read_csv(LOG_FILE)

    return jsonify({
        "total_predictions": len(logs),
        "avg_probability": float(logs["probability"].mean()),
        "promoted": int((logs["prediction"] == 1).sum()),
        "not_promoted": int((logs["prediction"] == 0).sum())
    })


# =========================================
# 9. HTML DASHBOARD
# =========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gender Equality MLOps Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background:#f4f6f9;padding:40px;">

<div class="container">

    <div class="card p-4">

        <h2>📊 Gender Equality ML Dashboard</h2>

        <hr>

        <p><b>Model Status:</b> {{ status }}</p>
        <p><b>Total Data:</b> {{ rows }}</p>

        <hr>

        <a href="/retrain" class="btn btn-primary">🚀 Manual Retrain</a>

        {% if msg %}
        <div class="alert alert-info mt-3">{{ msg }}</div>
        {% endif %}

    </div>

</div>

</body>
</html>
"""


# =========================================
# 10. DASHBOARD ROUTE
# =========================================
@app.route('/dashboard')
def dashboard():

    rows = 0
    if os.path.exists(DATA_PATH):
        rows = len(pd.read_csv(DATA_PATH))

    return render_template_string(HTML_TEMPLATE,
                                  status="ACTIVE",
                                  rows=rows,
                                  msg=None)


# =========================================
# 11. MANUAL RETRAIN
# =========================================
@app.route('/retrain')
def retrain():

    success = retrain_model()

    msg = "✅ Retraining Successful" if success else "❌ Retraining Failed"

    rows = len(pd.read_csv(DATA_PATH)) if os.path.exists(DATA_PATH) else 0

    return render_template_string(HTML_TEMPLATE,
                                  status="ACTIVE",
                                  rows=rows,
                                  msg=msg)


# =========================================
# 12. RUN APP
# =========================================
if __name__ == "__main__":
    print("🚀 App running at http://127.0.0.1:5000")
    app.run(debug=True)