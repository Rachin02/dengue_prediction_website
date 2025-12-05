import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

# Use the default 'templates' folder (we added templates/index.html)
flask_app = Flask(__name__, template_folder='templates')

# Load model safely and give an informative error if missing
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    # Keep model as None and log error — app will still start for debugging
    print(f"Failed to load model from {MODEL_PATH}: {e}")

# Determine the expected feature columns (same logic used during training)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Dengue_clinical_dataset.csv')
FEATURE_COLUMNS = []
try:
    df_full = pd.read_csv(CSV_PATH)
    df_model = df_full.drop(columns=['Id', 'Location'])
    bool_features = ['Fever', 'Headache', 'Muscle_Pain', 'Rash', 'Vomiting']
    # numeric features are everything else except Outcome and boolean features
    num_features = [col for col in df_model.columns if col not in ['Outcome'] + bool_features]
    FEATURE_COLUMNS = df_model.drop('Outcome', axis=1).columns.tolist()
except Exception as e:
    print(f"Failed to read CSV for feature columns: {e}")


@flask_app.route("/")
def Home():
    return render_template('index.html')


def to_numeric(val):
    """Convert common form inputs to numeric values used by the model.
    - yes/no -> 1/0
    - male/female -> 1/0 (male=1)
    - numeric strings -> float
    """
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ('yes', 'y', 'true', '1'):
            return 1.0
        if s in ('no', 'n', 'false', '0'):
            return 0.0
        if s in ('male', 'm'):
            return 1.0
        if s in ('female', 'f'):
            return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


@flask_app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded. Check server logs.')

    # Map form fields to the model's expected column names
    # Form field names -> CSV column names
    form_to_col = {
        'age': 'Age',
        'gender': 'Gender',
        'duration': 'Duration_of_Fever',
        'platelet': 'Platelet Count',
        'wbc': 'WBC',
        'fever': 'Fever',
        'rash': 'Rash',
        'vomiting': 'Vomiting',
        'headache': 'Headache',
        'muscle_pain': 'Muscle_Pain'
    }

    # Build a single-row DataFrame with the same columns and order used during training
    if not FEATURE_COLUMNS:
        return render_template('index.html', prediction_text='Server missing feature metadata; contact admin.')

    data_row = {col: 0 for col in FEATURE_COLUMNS}
    for form_key, form_val in request.form.items():
        target_col = form_to_col.get(form_key)
        if target_col and target_col in data_row:
            data_row[target_col] = to_numeric(form_val)

    try:
        features_df = pd.DataFrame([data_row], columns=FEATURE_COLUMNS)
        prediction = model.predict(features_df)
        # prediction may be an array like [0] or [1], or probabilities.
        raw = prediction[0]
        # Normalize to a single numeric value when possible
        try:
            # numpy types support float conversion
            val = float(raw)
        except Exception:
            try:
                val = int(raw)
            except Exception:
                val = raw

        label = None
        # If it's a numeric probability between 0 and 1, threshold at 0.5
        try:
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                label = 'Dengue Positive' if val >= 0.5 else 'Dengue Negative'
            else:
                # Otherwise treat as class label (0/1)
                intval = int(round(val))
                label = 'Dengue Positive' if intval == 1 else 'Dengue Negative'
        except Exception:
            # Fallback: show raw prediction
            label = f'The predicted value is {raw}'

        return render_template('index.html', prediction_text=label)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction failed: {e}')


if __name__ == "__main__":
    flask_app.run(debug=True)