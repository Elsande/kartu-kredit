from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model
model = joblib.load('naive_bayes_model.pkl')

# Define mappings for input conversion
mapping_pendapatan = {'<200000': 0, '200000-400000': 1, '>400000': 2}
mapping_status = {'Menikah': 0, 'Belum Menikah': 1, 'Cerai': 2}
mapping_tempat_tinggal = {'Punya': 0, 'Dengan Orang Tua': 1, 'Kontrak': 2}
mapping_mobil = {'Punya': 0, 'Tidak Punya': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract and convert the input data
        pendapatan = mapping_pendapatan[data.get('Pendapatan')]
        status = mapping_status[data.get('Status')]
        anak = int(data.get('Anak'))  # Assuming 'Anak' is provided as integer
        tempat_tinggal = mapping_tempat_tinggal[data.get('Tempat Tinggal')]
        mobil = mapping_mobil[data.get('Mobil')]

        # Prepare the feature array for prediction
        features = np.array([[pendapatan, status, anak, tempat_tinggal, mobil]])

        # Make the prediction using the loaded model
        prediction = model.predict(features)

        # Map the prediction to the corresponding class
        prediction_mapping = {0: 'Tidak Lolos', 1: 'Lolos'}
        result = prediction_mapping[prediction[0]]

        # Return the prediction result as a JSON response
        return jsonify(result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
