# deep-learning-tamil-inscription-recognition
Deep Learning based OCR system for recognizing ancient Tamil inscription characters from stone inscription images using preprocessing, character segmentation, CNN classification,  and a Flask web application. The system processes inscription images through multiple stages including preprocessing (noise removal and enhancement), character segmentation, and CNN-based character classification. The recognized characters are then displayed through an interactive Flask web application where users can upload inscription images and view predicted characters. This project demonstrates the application of OCR, image processing, and deep learning techniques for digital epigraphy and historical document analysis.



├── preprocessing.py        # Image preprocessing
├── segmentation.py         # Character segmentation
├── prediction.py           # CNN prediction
├── app.py                  # Flask web application
│
├── templates/
│   └── index.html          # Web interface
│
├── static/
│   └── uploads/            # Uploaded and segmented images
│
├── Model-Creation/
│   ├── Recognition_1.ipynb
│   ├── Recognition_2.ipynb
│   └── Recognition_3.ipynb
│
└── CNN.keras               # Trained CNN model
