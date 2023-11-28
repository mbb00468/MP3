from transformers import ViTImageProcessor, ViTForImageClassification
from flask import Flask, render_template, request, flash, redirect
from PIL import Image
from io import BytesIO
import base64
import os
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)  # Use a random secret key for security

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_vit(image):
    try:
        # Load the ViT model and processor outside the function
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Extract features (patches) from the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Predict by feeding the model
        outputs = model(**inputs)

        # Convert outputs to logits
        logits = outputs.logits

        # Model predicts one of the classes by picking the logit with the highest probability
        predicted_class_idx = logits.argmax(-1).item()

        predicted_class = model.config.id2label[predicted_class_idx]
        return predicted_class
    except Exception as e:
        return f"Error predicting with ViT: {str(e)}"

def predict_yolo(image):
    try:
        yolo_model = YOLO('yolov8n.pt')
        results = yolo_model(image, verbose=False)  # results list

        pred_classes = []

        for result in results:
            boxes = result.boxes.cpu().numpy()  # get boxes on the CPU in numpy
            for box in boxes:  # iterate boxes
                pred_classes.append(result.names[int(box.cls[0])])

        return pred_classes  # return the list of predicted classes
    except Exception as e:
        return f"Error predicting with YOLO: {str(e)}"

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            return render_template('model_choice.html', filename=file.filename)
        else:
            flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.')
            return redirect(request.url)
    else:
        return render_template('home.html')

@app.route('/model_choice', methods=['POST'])
def model_choice():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    model_choice = request.form.get('model_choice')

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        if model_choice == 'vit':
            result = predict_vit(file)
            return render_template('vit_result.html', result=result)

        elif model_choice == 'yolo':
            result = predict_yolo(file)
            return render_template('yolo_result.html', result=result)
        else:
            flash('Invalid model choice')
            return redirect(request.url)
    else:
        flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
