from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Inicializando a aplicação Flask
app = Flask(__name__)

# Configurar o Flask-CORS
CORS(app)

# Carregar o modelo treinado
model = joblib.load('modelo_randomforest.joblib')

# Função para fazer a previsão
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

@app.route('/predict', methods=['POST'])
def predictions():
    if request.method == 'POST':
        data = request.get_json()
        Age = data.get('Age')
        Pregnancies = data.get('Pregnancies')
        Glucose = data.get('Glucose')
        BloodPressure = data.get('BloodPressure')
        Insulin = data.get('Insulin')
        BMI = data.get('BMI')
        SkinThickness = data.get('SkinThickness')
        DPF = data.get('DPF')
        result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
        return jsonify(result)
    return "Invalid request method"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
