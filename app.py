from transformers import ViTImageProcessor, ViTForImageClassification
from flask import Flask, render_template, request, flash, redirect, url_for
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

        # Open the image and normalize pixel values
        image = Image.open(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)

        # Extract features (patches) from the image
        inputs = feature_extractor(images=image.unsqueeze(0))  # Add a batch dimension

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


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        file = request.files['file']
        # Save the file to your desired location
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        
        session['file_path'] = file.filename
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('home'))

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))

        if file and allowed_file(file.filename):
            # Save the uploaded file to the 'uploads' folder
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            return render_template('model_choice.html', filename=file.filename)
        else:
            flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.')
            return redirect(url_for('home'))

    else:
        return render_template('home.html')

@app.route('/model_choice', methods=['POST', 'GET'])
def model_choice():
    try:


        if file.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(url_for('home'))

        if file and allowed_file(file.filename):
            print(f"Model choice: {model_choice}")

            # Construct the path to the uploaded file
            upload_folder = app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, file.filename)

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
                return redirect(url_for('home'))

        else:
            flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.')
            print('Invalid file type. Allowed types are png, jpg, jpeg, gif.')
            return redirect(url_for('home'))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        flash('An error occurred. Please try again.')
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
