#Flask API za CIFAR-10 klasifikaciju

import os
import io
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

MODELS_DIR = Path('models')

#Cache za ucitane modele
_model_cache = {}

def load_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        return None

    print(f"Ucitavanje modela: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    _model_cache[model_name] = model
    return model

def preprocess_image(image_bytes):
    #Pretvaranje upload slike u 32x32 RGB numpy array normalizovan na [0,1]
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

@app.route('/predict', methods=['POST'])
def predict():
    #Validacija
    if 'image' not in request.files:
        return jsonify({'error': 'Nema slike u zahtevu'}), 400

    model_name = request.form.get('model', 'Model_1_Baseline')
    image_file = request.files['image']

    #Ucitavanje modela
    model = load_model(model_name)
    if model is None:
        return jsonify({'error': f'Model {model_name} nije pronadjen. Pokreni save_model.py prvo.'}), 404

    #Preprocessing
    image_bytes = image_file.read()
    img_array = preprocess_image(image_bytes)

    #Predikcija
    predictions = model.predict(img_array, verbose=0)[0]  # (10,)

    #Formatiranje rezultata
    results = [
        {'class': CLASSES[i], 'confidence': float(predictions[i])}
        for i in range(len(CLASSES))
    ]
    results.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({'predictions': results, 'model': model_name})

@app.route('/models', methods=['GET'])
def list_models():
    #Vracanje liste dostupnih modela
    available = []
    for model_name in ['Model_1_Baseline', 'Model_2_Deep', 'Model_3_Wide',
                       'Model_4_Small', 'Model_5_LargeKernel']:
        path = MODELS_DIR / f"{model_name}.keras"
        available.append({'name': model_name, 'available': path.exists()})
    return jsonify(available)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def frontend_files(filename):
    return send_from_directory('frontend', filename)

if __name__ == '__main__':
    print("=" * 50)
    print("CIFAR-10 Backend API")
    print("=" * 50)

    #Provera modela
    available = [f.stem for f in MODELS_DIR.glob('*.keras')] if MODELS_DIR.exists() else []
    if not available:
        print("UPOZORENJE: Nema sacuvanih modela!")
        print("Pokreni: python save_model.py")
    else:
        print(f"Dostupni modeli: {available}")

    print("\nServer pokrenut na: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

