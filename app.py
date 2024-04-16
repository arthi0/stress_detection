from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from main import extract_features

app = Flask(__name__)

# Load the model architecture from JSON file
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

# Load the model weights
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Route to render the home page
@app.route('/')
def home():
    return render_template('webpage\home.html')
# Route to handle image upload and predict emotion
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            try:
                # Save uploaded image to buffer
                img_path = 'temp_image.png'  # Temporary image save path
                file.save(img_path)
                img_array = extract_features(img_path)
                pred = model.predict(img_array)
                pred_label = label[np.argmax(pred)]
                return render_template('webpage\result.html', prediction=pred_label)
            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
