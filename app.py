from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Create the uploads directory if it does not exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
crop_model = joblib.load('static/model/crop_model.pkl')
fertilizer_model = joblib.load('static/model/fertilizer_model.pkl')
disease_model = load_model('static/model/Disease_Detection.h5')

# Define the classes for disease detection
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Define the prevention methods for each disease
prevention_methods_dict = {
    'Apple___Apple_scab': ["Prune infected leaves", "Apply appropriate fungicides"],
    'Apple___Black_rot': ["Remove and destroy infected fruit", "Apply fungicides during bloom"],
    'Apple___Cedar_apple_rust': ["Remove nearby juniper trees", "Apply fungicides in spring"],
    'Apple___healthy': ["Maintain regular tree care", "Monitor for signs of disease"],
    'Blueberry___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Cherry_(including_sour)___Powdery_mildew': ["Apply fungicides", "Prune affected areas"],
    'Cherry_(including_sour)___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ["Rotate crops", "Use resistant hybrids"],
    'Corn_(maize)___Common_rust_': ["Use rust-resistant hybrids", "Apply fungicides if necessary"],
    'Corn_(maize)___Northern_Leaf_Blight': ["Plant resistant varieties", "Rotate crops"],
    'Corn_(maize)___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Grape___Black_rot': ["Prune and remove infected leaves", "Apply fungicides"],
    'Grape___Esca_(Black_Measles)': ["Remove and destroy infected vines", "Apply fungicides if necessary"],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ["Improve air circulation by pruning", "Apply fungicides"],
    'Grape___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Orange___Haunglongbing_(Citrus_greening)': ["Control psyllid populations", "Remove and destroy infected trees"],
    'Peach___Bacterial_spot': ["Apply bactericides", "Plant resistant varieties"],
    'Peach___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Pepper,_bell___Bacterial_spot': ["Use disease-free seeds", "Apply copper-based bactericides"],
    'Pepper,_bell___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Potato___Early_blight': ["Apply fungicides", "Rotate crops"],
    'Potato___Late_blight': ["Use resistant varieties", "Apply fungicides"],
    'Potato___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Raspberry___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Soybean___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Squash___Powdery_mildew': ["Apply fungicides", "Ensure good air circulation"],
    'Strawberry___Leaf_scorch': ["Apply fungicides", "Remove and destroy infected leaves"],
    'Strawberry___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"],
    'Tomato___Bacterial_spot': ["Use disease-free seeds", "Apply copper-based bactericides"],
    'Tomato___Early_blight': ["Rotate crops", "Apply fungicides"],
    'Tomato___Late_blight': ["Use resistant varieties", "Apply fungicides"],
    'Tomato___Leaf_Mold': ["Improve air circulation", "Apply fungicides"],
    'Tomato___Septoria_leaf_spot': ["Remove infected leaves", "Apply fungicides"],
    'Tomato___Spider_mites Two-spotted_spider_mite': ["Use miticides", "Increase humidity around plants"],
    'Tomato___Target_Spot': ["Apply fungicides", "Remove and destroy infected leaves"],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ["Control whitefly populations", "Use resistant varieties"],
    'Tomato___Tomato_mosaic_virus': ["Use virus-free seeds", "Practice crop rotation"],
    'Tomato___healthy': ["Ensure proper watering and sunlight", "Monitor for signs of disease"]
}



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop')
def crop_recommendation():
    return render_template('crop_recommendation.html')


@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')

@app.route('/disease-detection')
def disease_detection():
    return render_template('disease_detection.html')


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
        values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]

        if Ph > 0 and Ph <= 14 and Temperature < 100 and Humidity > 0:
            # Predict using the model
            prediction = crop_model.predict([values])

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
        input_data = pd.DataFrame([[Nitrogen, Potassium, Phosphorous]], columns=['Nitrogen', 'Potassium', 'Phosphorous'])
        result = fertilizer_model.predict(input_data)[0]

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


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    return classes[np.argmax(predictions)]

@app.route('/predict', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = model_predict(file_path, disease_model)
            os.remove(file_path)

            prevention_methods = prevention_methods_dict.get(prediction, [])

            return render_template('disease_detection.html', result=prediction, prevention_methods=prevention_methods)
        return render_template('disease_detection.html', error='No file selected or file is empty')
    except Exception as e:
        return render_template('disease_detection.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run()
