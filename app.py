import os
from flask import Flask, request, jsonify
from numpy import random

app = Flask(__name__)
from inference import get_skin_prediction

@app.route('/', methods=['GET'])
def basic_get():
    return jsonify({'endpoints': {'POST/upload': 'takes jpeg via multipart form with key of file'}})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file in request'}), 400
    
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    file=request.files['file']
    image= file.read()
    diagnosis=get_skin_prediction(image)   
    return jsonify({'type': diagnosis}), 200

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 3030))

