from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop')
def crop_recommendation():
    return render_template('crop_recommendation.html')


@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')


@app.route('/crop_predict', methods=["POST"])
def crop_predict():
    try:
        # Get form data
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])

        # Prepare data for prediction
        values = [Nitrogen, Phosphorus, Potassium,
                  Temperature, Humidity, Ph, Rainfall]
        model = joblib.load('static/model/crop_model.pkl')

        if Ph > 0 and Ph <= 14 and Temperature < 100 and Humidity > 0:
            # Load the model
            model = joblib.load('static/model/crop_model.pkl')

            # Predict using the model
            prediction = model.predict([values])

            # Return prediction as JSON
            return jsonify({'prediction': str(prediction[0])})
        else:
            return jsonify({'error': 'Error in entered values in the form. Please check the values and fill it again.'})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})


@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    try:
        # Get form data
        Nitrogen = float(request.form.get('Nitrogen', 0))
        Potassium = float(request.form.get('Potassium', 0))
        Phosphorous = float(request.form.get('Phosphorous', 0))

        # Prepare data for prediction
        input_data = pd.DataFrame([[Nitrogen, Potassium, Phosphorous]],
                                  columns=['Nitrogen', 'Potassium', 'Phosphorous'])
        model = joblib.load('static/model/fertilizer_model.pkl')
        result = model.predict(input_data)[0]

        # Map result to the corresponding recommendation
        recommendations = {
            0: 'TEN-TWENTY SIX-TWENTY SIX',
            1: 'Fourteen-Thirty Five-Fourteen',
            2: 'Seventeen-Seventeen-Seventeen',
            3: 'TWENTY-TWENTY',
            4: 'TWENTY EIGHT-TWENTY EIGHT',
            5: 'DAP'
        }
        result = recommendations.get(result, 'UREA')

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('fertilizer_recommendation.html', result=str(result))


if __name__ == '__main__':
    app.run(port=8501)
