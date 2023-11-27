from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import time as timer  # Rename 'time' import to 'timer'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Define custom confidence metric
def confidence(y_true, y_pred):
    return tf.reduce_max(y_pred, axis=-1)

# Load model with custom_objects to include the custom metric
model_path = 'MobileNet_Model.h5'
with tf.keras.utils.custom_object_scope({'confidence': confidence}):
    model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            
            # Preprocess the uploaded image
            processed_img = preprocess_image(file_path)
            
            # Start time processing
            start_time = timer.time()

            # Make prediction
            prediction = model.predict(processed_img)
            class_idx = np.argmax(prediction)
            
            if class_idx == 0:
                class_name = 'Flood'
            else:
                class_name = 'Non-Flood'
            
            confidence_score = prediction[0][class_idx]

            # End time processing
            end_time = timer.time()

            # Calculate processing time
            processing_time = end_time - start_time

            return render_template('result.html', image_path=file_path, class_name=class_name, confidence_score=confidence_score, processing_time=f"{processing_time:.2f} seconds")

if __name__ == '__main__':
    app.run(debug=True)
