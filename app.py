import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the fruit classification model
fruit_model = load_model('fruit_classifier_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))  # Adjust the target size to match your model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Define a dictionary to map class indices to fruit names
class_indices_to_fruits = {
    0: 'apple',
    1: 'banana',
    2: 'orange'
}

@app.route('/')
def index():
    return render_template('upload.html', prediction=None, freshness=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            processed_image = preprocess_image(image_path)
            fruit_class = np.argmax(fruit_model.predict(processed_image), axis=1)[0]
            predicted_fruit = class_indices_to_fruits[fruit_class]

            # Load the corresponding classification model
            classification_model_path = f'{predicted_fruit}_classification_model.h5'
            classification_model = load_model(classification_model_path)

            # Perform classification using the selected model
            classification_result = classification_model.predict(processed_image)

            fresh_probability = classification_result[0][0] * 100
            not_fresh_probability = 100 - fresh_probability  

            # Round the freshness percentages to two decimal places
            fresh_probability = round(fresh_probability, 4)
            not_fresh_probability = round(not_fresh_probability, 4)

            return render_template('upload.html', prediction=predicted_fruit.capitalize(), freshness=fresh_probability, not_freshness=not_fresh_probability, error=None)
        else:
            return render_template('upload.html', prediction=None, freshness=None, not_freshness=None, error='No file selected. Please choose an image to upload.')

    except Exception as e:
        return render_template('upload.html', prediction=None, freshness=None, not_freshness=None, error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
