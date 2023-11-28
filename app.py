from flask import Flask, request, session, redirect, url_for, render_template

import os
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
from ultralytics import YOLO


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"  # for flashing messages

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # If the user does not select a file, the browser also submits an empty part without a filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Ensure a safe filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['file_path'] = file_path  # Save the full file path in the session
        print('File successfully uploaded and saved!')
        return render_template('model_choice.html')

    flash('Invalid file type. Please upload an image.')
    return redirect(request.url)


@app.route('/model_choice', methods=['GET', 'POST'])
def model_choice():
    try:
        if request.method == 'POST':
            model_choice = request.form.get('model_choice')

            # Get the file path from the session or form data
            file_path = session.get('file_path')

            if not file_path:
                print('File path not found. Please upload a file.')
                return redirect(url_for('index'))

            if model_choice == 'vit':
                result = predict_vit(file_path)
                return render_template('vit_result.html', result=result)

            elif model_choice == 'yolo':
                result_image_path= predict_yolo(file_path)
                return render_template('yolo_result.html', result_image=result_image_path)

    except KeyError:
        print('File path not found in the session. Please upload a file.')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print('An error occurred. Please try again.')
        return render_template('model_choice.html')


def predict_vit(file_path):
    try:
        # Load the ViT model and processor outside the function
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Open the image and normalize pixel values
        image = Image.open(file_path)
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Predict by feeding the model
        outputs = model(**inputs)

        # Convert outputs to logits
        logits = outputs.logits

        # Model predicts one of the classes by picking the logit with the highest probability
        predicted_class_idx = torch.argmax(logits, dim=1).item()

        # In this example, we assume model.config.id2label is available
        predicted_class = model.config.id2label[predicted_class_idx]
        return predicted_class
    except Exception as e:
        return f"Error predicting with ViT: {str(e)}"


def predict_yolo(file_path):
    try:
        # Define the predict_yolo function
        model = YOLO('yolov8n.pt')
        results = model(file_path)

        # Extract bounding boxes and labels
        bounding_boxes = []
        for result in results:
            for box in result.boxes.xyxy:
                # Ensure that there are at least 6 elements in the box array before accessing index 5
                if len(box) >= 6:
                    bounding_boxes.append({
                        'left': box[0].item(),
                        'top': box[1].item(),
                        'right': box[2].item(),
                        'bottom': box[3].item(),
                        'label': model.names[int(box[5].item())],
                    })

        # Display and save the results
        for i, result in enumerate(results):
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])
            result_image_path = os.path.join(UPLOAD_FOLDER, f'result_{i}.jpg')
            im.save(result_image_path)
            print(result_image_path)

        print(f"YOLO prediction result: {bounding_boxes}")
        return result_image_path
    except Exception as e:
        return f"Error predicting with YOLO: {str(e)}", []

# The rest of your code remains unchanged




if __name__ == '__main__':
    app.run(debug=True)
