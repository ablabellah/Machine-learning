from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Charger le modèle et les données
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = float(request.form.get('kilo_driven'))

    # Préparer les données pour la prédiction
    input_data = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [driven],
        'fuel_type': [fuel_type]
    })

    # Effectuer la prédiction
    prediction = model.predict(input_data)
    prediction_value = np.round(prediction[0], 2)

    return str(prediction_value)

if __name__ == '__main__':
    app.run(debug=True)
