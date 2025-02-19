# Flask Application README

# Project Title: Flask Image Processing Application

## Description

This project is a Flask application that processes images using a pre-trained TensorFlow model. It allows users to upload images, predicts labels based on the model, and scrapes product information from a specified website.

## Project Structure

```
flask-app
├── Dockerfile
├── requirements.txt
├── src
│   ├── backend.py
└── README.md
```

## Requirements

- Docker
- Python 3.x

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd flask-app
   ```

2. Build the Docker image:
   ```
   docker build -t flask-app .
   ```

## Running the Application

To run the application, use the following command:

```
docker run -p 5000:5000 flask-app
```

The application will be accessible at `http://localhost:5000/upload`.

## Usage

To upload an image, send a POST request to the `/upload` endpoint with the image file included in the request. The application will return the predicted label and scraped product information.

## License

This project is licensed under the MIT License.
