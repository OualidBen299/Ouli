import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
import io
import easyocr

os.environ['LMDB_PURE'] = '1'

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(['en'], gpu=False)

def decode_base64_image(base64_image_url):
    try:
        image_data = base64.b64decode(base64_image_url.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_base64_image(image):
    _, buffer = cv2.imencode('.png', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_image}"

def unsharp_filter(image):
    pil_image = Image.fromarray(image.astype('uint8'))
    new_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return np.array(new_image)

def extract_numbers(image):
    try:
        results = reader.readtext(image)
        if results is None:
            return None
        numbers = ''.join([result[1] for result in results if result[1].isdigit()])
        return numbers if len(numbers) == 3 else None
    except Exception as e:
        print(f"Error extracting numbers: {e}")
        return None

@app.route('/CrossCaptcha', methods=['POST'])
def ocr_route():
    data = request.get_json()
    base64_image_url = data.get('base64ImageUrl')
    if not base64_image_url:
        return jsonify({"error": "No image URL provided"}), 400
    image = decode_base64_image(base64_image_url)
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400
    numbers = extract_numbers(image)
    if numbers:
        return jsonify({"solution": numbers})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numbers = extract_numbers(gray)
    if numbers:
        return jsonify({"solution": numbers})
    unsharped = unsharp_filter(gray)
    numbers = extract_numbers(unsharped)
    if numbers:
        return jsonify({"solution": numbers})
    return jsonify({"error": "Failed to extract numbers from the image"}), 400

if __name__ == '__main__':
    app.run(debug=True)
