# app.py

from flask import Flask, render_template, request
import os

# Import your exact modules
from preprocessing import preprocess_image
from segmentation import segment_characters
from prediction import predict_character


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def home():

    predictions = []
    final_text = ""

    if request.method == 'POST':

        # Save uploaded image
        file = request.files['image']
        original_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(original_path)

        # -----------------------------
        # 1️⃣ FULL PREPROCESSING
        # -----------------------------
        processed_path = os.path.join(UPLOAD_FOLDER, "ImagePreProcessingFinal.jpg")
        preprocess_image(original_path, processed_path)

        # -----------------------------
        # 2️⃣ SEGMENTATION
        # -----------------------------
        roi_paths = segment_characters(processed_path, UPLOAD_FOLDER)

        # -----------------------------
        # 3️⃣ PREDICTION
        # -----------------------------
        for roi in roi_paths:

            label = predict_character(roi)

            final_text += label

            predictions.append({
                "image": os.path.basename(roi),
                "label": label
            })

    return render_template("index.html",
                           predictions=predictions,
                           final_text=final_text)


if __name__ == "__main__":
    app.run(debug=True)
