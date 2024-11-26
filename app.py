from flask import Flask, request, jsonify
import joblib

# Inicializando a aplicação Flask
app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load('modelo_randomforest.joblib')

# Função para fazer a previsão
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    # Organizar os dados em um formato adequado para o modelo (uma lista ou array)
    features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
    
    # Fazer a previsão
    prediction = model.predict(features)
    
    # Retornar o resultado
    return {"prediction": int(prediction[0])}

@app.route('/predict', methods=['POST'])
def predictions():
    if request.method == 'POST':
        # Receber os dados JSON da requisição
        data = request.get_json()
        
        # Obter os valores das variáveis
        Age = data.get('Age')
        Pregnancies = data.get('Pregnancies')
        Glucose = data.get('Glucose')
        BloodPressure = data.get('BloodPressure')
        Insulin = data.get('Insulin')
        BMI = data.get('BMI')
        SkinThickness = data.get('SkinThickness')
        DPF = data.get('DPF')
        
        # Chamar a função de previsão com os dados recebidos
        result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
        
        # Retornar o resultado como JSON
        return jsonify(result)
    
    return "Invalid request method"

if __name__ == '__main__':
    # Rodar o servidor Flask na porta 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
