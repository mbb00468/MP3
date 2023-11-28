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

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"  # for flashing messages

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)

    file = request.files['file']

    
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)

    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Ensure a safe filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['file_path'] = file_path  # Save the full file path in the session
        print('File successfully uploaded and saved!')
        return render_template('model_choice.html')

    print('Invalid file type. Please upload an image.')
    return redirect(request.url)


@app.route('/model_choice', methods=['GET', 'POST'])
def model_choice():
    try:
        if request.method == 'POST':
            model_choice = request.form.get('model_choice')
            
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
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        image = Image.open(file_path)
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        
        predicted_class = model.config.id2label[predicted_class_idx]
        return predicted_class
    except Exception as e:
        return f"Error predicting with ViT: {str(e)}"


def predict_yolo(file_path):
    try:
        
        model = YOLO('yolov8n.pt')
        results = model(file_path)

        bounding_boxes = []
        for result in results:
            for box in result.boxes.xyxy:
                
                if len(box) >= 6:
                    bounding_boxes.append({
                        'left': box[0].item(),
                        'top': box[1].item(),
                        'right': box[2].item(),
                        'bottom': box[3].item(),
                        'label': model.names[int(box[5].item())],
                    })

        
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



if __name__ == '__main__':
    app.run(debug=True)
