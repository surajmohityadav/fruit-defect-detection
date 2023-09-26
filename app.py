import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Define model paths for each fruit
model_paths = {
    'apple': 'apple_classification_model.h5',
    'banana': 'banana_classification_model.h5',
    'orange': 'orange_classification_model.h5'
}

# Load models for each fruit
models = {fruit: load_model(path) for fruit, path in model_paths.items()}

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('upload.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['image']
        selected_fruit = request.form['fruit']  # Get the selected fruit option

        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            processed_image = preprocess_image(image_path)
            prediction = models[selected_fruit].predict(processed_image)

            if prediction[0][0] > 0.5:
                result = 'Rotten'
            else:
                result = 'Fresh'

            return render_template('upload.html', prediction=result, error=None)
        else:
            return render_template('upload.html', prediction=None, error='No file selected. Please choose an image to upload.')

    except Exception as e:
        return render_template('upload.html', prediction=None, error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
