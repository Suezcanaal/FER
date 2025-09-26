from flask import Flask, request, jsonify
from fer import FER
import cv2
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
detector = FER(mtcnn=True)  # model for emotion recognition

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']  # base64 encoded image from frontend
        img_data = base64.b64decode(data.split(',')[1])  
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Detect emotions
        results = detector.detect_emotions(frame)

        if results:
            top_emotion, score = detector.top_emotion(frame)
            return jsonify({"emotion": top_emotion, "score": score, "details": results})
        else:
            return jsonify({"error": "No face detected"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
