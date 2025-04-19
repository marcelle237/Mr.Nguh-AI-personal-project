import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.h5')
CLASS_NAMES = sorted(os.listdir(os.path.join(os.path.dirname(__file__), '..', 'animals10', 'raw-img')))
# Italian to English mapping for Animals-10
CLASS_NAME_MAP = {
    'cane': 'dog',
    'cavallo': 'horse',
    'elefante': 'elephant',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'gatto': 'cat',
    'mucca': 'cow',
    'pecora': 'sheep',
    'ragno': 'spider',
    'scoiattolo': 'squirrel'
}
IMG_SIZE = (224, 224)

# Flask app
app = Flask(__name__)
model = load_model(MODEL_PATH)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name_it = CLASS_NAMES[class_idx]
    class_name_en = CLASS_NAME_MAP.get(class_name_it, class_name_it)
    confidence = float(np.max(preds))
    return class_name_en, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join('static', 'uploads', file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            pred, conf = model_predict(filepath)
            prediction = pred
            confidence = f"{conf*100:.2f}%"
            return render_template('index.html', prediction=prediction, confidence=confidence, img_url=filepath)
    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
