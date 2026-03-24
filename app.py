from flask import Flask, render_template, request, jsonify
from fer import FER
import cv2
import numpy as np
import io

app = Flask(__name__)

# create detector once
detector = FER()

def read_image_from_file(file_storage):
    data = file_storage.read()
    img_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # BGR -> RGB for FER
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/camera', methods=['GET'])
def camera_page():
    return render_template('camera.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    # support file upload (form-data with "image")
    file = request.files.get('image')
    if not file or file.filename == '':
        return "No file uploaded", 400

    img = read_image_from_file(file)
    if img is None:
        return "Invalid image", 400

    results = detector.detect_emotions(img)
    faces = []
    for r in results:
        box = r.get('box', [])
        emotions = r.get('emotions', {})
        top = max(emotions, key=emotions.get) if emotions else None
        faces.append({
            'box': box,
            'emotions': emotions,
            'top_emotion': top
        })

    # if request expects JSON (camera uses fetch), return JSON
    if request.headers.get('Accept') == 'application/json' or request.is_json:
        return jsonify({'faces': faces})

    # otherwise render template (file upload form)
    emotion_text = faces[0]['top_emotion'] if faces else 'No face detected'
    return render_template('index.html', emotion=emotion_text)

if __name__ == '__main__':
    app.run(debug=True)
