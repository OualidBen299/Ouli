import logging
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image, ImageFilter
import io
from paddleocr import PaddleOCR
from collections import deque

logging.getLogger('werkzeug').disabled = True
logging.getLogger('paddle').disabled = True
logging.getLogger('ppocr').disabled = True  # Disable PaddleOCR logs

app = Flask(__name__)
CORS(app)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def hide_output_after_start():
    sys.stdout = open('NUL', 'w') if sys.platform == 'win32' else open('/dev/null', 'w')

print(" * Starting Of Cross Captcha v1.0")
print(f" * Captcha Cross v1.0 is: \033[92mON\033[0m")
print(f" * If you want to stop Cross Captcha click : \033[91mCTRL + C\033[0m")
hide_output_after_start()

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

def bfs(visited, array, node):
    def getNeighboor(array, node):
        neighboors = []
        if node[0]+1 < array.shape[0]:
            if array[node[0]+1,node[1]] == 0:
                neighboors.append((node[0]+1,node[1]))
        if node[0]-1 >= 0:
            if array[node[0]-1,node[1]] == 0:
                neighboors.append((node[0]-1,node[1]))
        if node[1]+1 < array.shape[1]:
            if array[node[0],node[1]+1] == 0:
                neighboors.append((node[0],node[1]+1))
        if node[1]-1 >= 0:
            if array[node[0],node[1]-1] == 0:
                neighboors.append((node[0],node[1]-1))
        return neighboors

    queue = deque([node])
    visited.add(node)
    while queue:
        current_node = queue.popleft()
        for neighboor in getNeighboor(array, current_node):
            if neighboor not in visited:
                visited.add(neighboor)
                queue.append(neighboor)

def removeIsland(img_arr, threshold):
    visited = set()
    while 0 in img_arr:
        x, y = np.where(img_arr == 0)
        point = (x[0], y[0])
        bfs(visited, img_arr, point)

        if len(visited) <= threshold:
            for i in visited:
                img_arr[i[0], i[1]] = 1
        else:
            for i in visited:
                img_arr[i[0], i[1]] = 2
        visited.clear()
    img_arr = np.where(img_arr == 2, 0, img_arr)
    return img_arr

def extract_numbers(image):
    try:
        results = ocr.ocr(image, cls=True)
        if results is None:
            return None
        numbers = ''.join([result[1][0] for line in results for result in line if result[1][0].isdigit()])
        return numbers if len(numbers) == 3 else None
    except Exception as e:
        print(f"Error extracting numbers: {e}")
        return None

def check_for_solution(image, step_name, steps):
    steps[step_name] = encode_base64_image(image)
    numbers = extract_numbers(image)
    if numbers:
        return numbers, step_name, steps
    return None, None, steps

@app.route('/CrossCaptcha', methods=['POST'])
def ocr_route():
    data = request.get_json()
    base64_image_url = data.get('base64ImageUrl')
    if not base64_image_url:
        return jsonify({"error": "No image URL provided"}), 400

    image = decode_base64_image(base64_image_url)
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    steps = {}
    numbers, step_name, steps = check_for_solution(image, "original_image", steps)
    if numbers:
        return jsonify({"solution": numbers})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numbers, step_name, steps = check_for_solution(gray, "grayscale", steps)
    if numbers:
        return jsonify({"solution": numbers})

    unsharped = unsharp_filter(gray)
    numbers, step_name, steps = check_for_solution(unsharped, "unsharp_filter", steps)
    if numbers:
        return jsonify({"solution": numbers})

    median_filtered = cv2.medianBlur(unsharped, 3)
    numbers, step_name, steps = check_for_solution(median_filtered, "median_filter", steps)
    if numbers:
        return jsonify({"solution": numbers})

    thresholded = np.where(median_filtered > 195, 1, 0).astype(np.uint8) * 255
    numbers, step_name, steps = check_for_solution(thresholded, "thresholding", steps)
    if numbers:
        return jsonify({"solution": numbers})

    island_removed = removeIsland(thresholded // 255, 30) * 255
    numbers, step_name, steps = check_for_solution(island_removed, "remove_islands", steps)
    if numbers:
        return jsonify({"solution": numbers})

    final_processed = cv2.medianBlur(island_removed, 3)
    numbers, step_name, steps = check_for_solution(final_processed, "final_median_filter", steps)
    if numbers:
        return jsonify({"solution": numbers})

    return jsonify({"error": "Failed to extract numbers from the image", "steps": steps}), 400

if __name__ == '__main__':
    app.run(debug=True)
