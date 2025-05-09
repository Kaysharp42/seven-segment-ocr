import os
import cv2
import numpy as np
import string
import tensorflow as tf
import pymysql
import argparse
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import the prediction function from predict.py
from predict import prepare_input, predict

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model_float16.tflite',
                    help='Path to the TFLite model file')
args = parser.parse_args()

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "1234"
DB_NAME = "water_meter"

# Get alphabet from predict.py
alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

# Get absolute path to the model
MODEL_PATH = os.path.abspath(args.model)
if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: Model file not found at '{MODEL_PATH}'")
    print(f"Current working directory: {os.getcwd()}")
    print("Available models: ")
    for file in os.listdir():
        if file.endswith(".tflite"):
            print(f" - {file}")
    print("Please specify the correct model path using --model argument")
    exit(1)
else:
    print(f"Using model at: {MODEL_PATH}")

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db_connection():
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection, True
    except Exception as e:
        print(f"Database connection error: {e}")
        return None, False

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Process the image using seven-segment OCR
    try:
        result = predict(file_path, MODEL_PATH)
        reading = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])
        
        # For detailed response, we can include individual digits and confidence
        # This is a placeholder - actual confidence might be calculated differently
        digits = [alphabet[index] for index in result[0] if index not in [blank_index, -1]]
        confidence_values = [0.95] * len(digits)  # Placeholder confidence values
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        # Save result into database
        connection, db_connected = get_db_connection()
        
        # Insert into database if connected
        if db_connected:
            try:
                with connection.cursor() as cursor:
                    sql = "INSERT INTO readings (reading, timestamp) VALUES (%s, %s)"
                    cursor.execute(sql, (reading, datetime.now()))
                    connection.commit()
            except Exception as e:
                connection.close()
                return jsonify({
                    'error': str(e), 
                    'reading': reading,
                    'confidence': avg_confidence
                }), 500
            finally:
                connection.close()

        return jsonify({
            'message': 'Image processed successfully', 
            'reading': reading,
            'digits': digits,
            'confidence': avg_confidence
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
