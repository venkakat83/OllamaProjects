from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import requests
from bs4 import BeautifulSoup

# Load your pre-trained model
model = load_model('../../../models/SchneiderProductsTrainingSchneiderProducts.h5')

label_map = {
    '0': 'A9D31616',
    '1': 'LTMR08PFM',
    '2': 'LV426100',
    '3': 'M9F11206',
    '4': 'S520057',
    '5': 'TPRST009',
    '6': 'XB4BA31'
}

app = Flask(__name__)
CORS(app)

def process_image(file):

        # Read the image file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image as required by your model
    image = cv2.resize(image, (224, 224))  # Example resize, adjust as needed
    image = image.astype('float32') / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    # Assuming the model returns a label
    label = np.argmax(predictions, axis=1)[0]
    # Get the prediction accuracy
    accuracy = np.max(predictions, axis=1)[0]
    print(accuracy)
    return str(label)

def scrape_website(label_name):
    url_map = {
        'A9D31616': 'https://www.se.com/in/en/product/A9D31616',
        'LTMR08PFM': 'https://www.se.com/in/en/product/LTMR08PFM',
        'LV426100': 'https://www.se.com/in/en/product/LV426100',
        'M9F11206': 'https://www.se.com/in/en/product/M9F11206',
        'S520057': 'https://www.se.com/fr/fr/product/S520057',
        'TPRST009': 'https://www.se.com/in/en/product/TPRST009',
        'XB4BA31': 'https://www.se.com/in/en/product/XB4BA31'
    }
    url = url_map.get(label_name)
    if not url:
        return 'No URL found for the given label'

    response = requests.get(url)
    if response.status_code != 200:
        return 'Failed to retrieve the webpage'

    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract relevant information from the webpage
    # Example: Extract the product description
    description = soup.find('meta', {'name': 'description'})
    if description:
        return description.get('content')
    return 'No description found'

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Request received")
    if 'image' not in request.files:
        print("No Image Part")
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        print("No Selected File")
        return jsonify({'error': 'No selected file'}), 400
    # Process the image file here
    label_index = process_image(file)
    label_name = label_map.get(label_index, 'Unknown')
    print(label_name)
    scraped_info = scrape_website(label_name)
    print(scraped_info)
    return jsonify({'message': 'Image processed successfully', 'label': label_name, 'info': scraped_info}), 200

if __name__ == '__main__':
    app.run(debug=True)