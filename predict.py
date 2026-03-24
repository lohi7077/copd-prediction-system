from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load models
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('copd_cnn_model.h5')

# Feature names in correct order (after dropping columns)
feature_names = [
    'AGE', 'PackHistory', 'MWT1Best', 'FEV1', 'FEV1PRED', 'FVC',
    'FVCPRED', 'CAT', 'HAD', 'SGRQ', 'AGEquartiles', 'gender',
    'smoking', 'Diabetes', 'muscular', 'hypertension', 'AtrialFib', 'IHD'
]

# COPD severity mapping
severity_map = {
    0: "No COPD",
    1: "Mild COPD",
    2: "Moderate COPD",
    3: "Severe COPD",
    4: "Very Severe COPD"
}


@app.route('/')
def home():
    return render_template('Predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form

    # Create input array in correct order
    input_values = [
        float(form_data['age']),
        float(form_data['pack_history']),
        float(form_data['mwt1_best']),
        float(form_data['fev1']),
        float(form_data['fev1_pred']),
        float(form_data['fvc']),
        float(form_data['fvc_pred']),
        float(form_data['cat']),
        float(form_data['had']),
        float(form_data['sgrq']),
        float(form_data['age_quartiles']),
        int(form_data['gender']),
        int(form_data['smoking']),
        int(form_data['diabetes']),
        int(form_data['muscle']),
        int(form_data['hypertension']),
        int(form_data['atrial_fib']),
        int(form_data['ihd'])
    ]

    # Create DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Scale features
    scaled_data = scaler.transform(input_df)

    # Reshape for CNN
    cnn_input = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)

    # Make prediction
    prediction = model.predict(cnn_input)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Get severity description
    severity = severity_map.get(pred_class, "Unknown")

    # Prepare probabilities
    probabilities = {
        severity_map[i]: f"{float(p) * 100:.1f}%"
        for i, p in enumerate(prediction[0])
    }

    return render_template('result.html',
                           prediction=severity,
                           confidence=f"{confidence * 100:.1f}%",
                           probabilities=probabilities)






if __name__ == '__main__':
    app.run(debug=True)





'''import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved CNN model and scaler
model = load_model('copd_cnn_model.h5')
scaler = joblib.load('scaler.pkl')

label_mapping = {
    'MILD': 0,
    'MODERATE': 1,
    'SEVERE': 2,
    'VERY SEVERE': 3
}
# Example: Replace this with your own input values in the same feature order used for training
# Ensure the number and order of features match exactly what the model was trained on.
sample_input = np.array([[71,3,420,1.67,93,2.58,118,21,1,29.29,3,0,2,0,0,0,0,0]])

# Scale the input
sample_input_scaled = scaler.transform(sample_input)

# Reshape for CNN: (samples, features, 1)
sample_input_cnn = sample_input_scaled.reshape(sample_input_scaled.shape[0], sample_input_scaled.shape[1], 1)

# Predict
prediction = model.predict(sample_input_cnn)

reverse_mapping = {v: k for k, v in label_mapping.items()}

predicted_class = np.argmax(model.predict(sample_input_cnn), axis=1)[0]
print(predicted_class)
predicted_severity = reverse_mapping[predicted_class]

print(f"Predicted COPD Severity: {predicted_severity}")
'''