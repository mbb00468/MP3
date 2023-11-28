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

    # If user does not select file, browser also submits an empty part without filename
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
            file_path = session['file_path']
        

            if model_choice == 'vit':
                result = predict_vit(file_path)
                print(f"ViT prediction result: {result}")
                return render_template('vit_result.html', result=result)

            elif model_choice == 'yolo':
                result = predict_yolo(file_path)
                print(f"YOLO prediction result: {result}")
                return render_template('yolo_result.html', result=result)

            else:
                flash('Invalid model choice')
                print('Invalid model choice')
                return redirect(url_for('model_choice'))
        print(request.files)
        print(request.form)
        
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

def predict_yolo(file):
    try:
        yolo_model = 'yolov8n.pt'
        results = yolo_model(file, verbose=False)  # results list

        pred_classes = []

        for result in results:
            boxes = result.boxes.cpu().numpy()  # get boxes on the CPU in numpy
            for box in boxes:  # iterate boxes
                pred_classes.append(result.names[int(box.cls[0])])

        return pred_classes  # return the list of predicted classes
    except Exception as e:
        return f"Error predicting with YOLO: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
