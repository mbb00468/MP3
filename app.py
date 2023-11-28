from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from flask import Flask, render_template, request
import os
from flask import flash, redirect
from ultralytics import YOLO
from PIL import Image
import warnings
import io
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"  # for flashing messages

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



def predict_vit(image):
    # Load the ViT model and processor
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    try:
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
    yolo_model = YOLO('yolov8n.pt')
    
    try:
        results = yolo_model(image, verbose=False)  # results list

        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            im.save('static/results.png')  # save image

        predclass = []

        for result in results:  # iterate results
            boxes = result.boxes.cpu().numpy()  # get boxes on the CPU in numpy
            for box in boxes:  # iterate boxes
                predclass.append(result.names[int(box.cls[0])])

        return predclass  # return the list of predicted classes

    except Exception as e:
        return f"Error predicting with YOLO: {str(e)}"


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')

        file = request.files['file']

        return render_template('model_choice.html', filename=file.filename)
    else:
        # Render the HTML form for GET requests
        return render_template('home.html')


@app.route('/model_choice', methods=['POST'])
def model_choice():
    model_choice = request.form.get('model_choice')
    file = request.files['file']

    if model_choice == 'vit':
        image = Image.open(file)
        result = predict_vit(image)
        return render_template('vit_result.html', result=result)

    elif model_choice == 'yolo':
        result = predict_yolo(file)
        return render_template('yolo_result.html', result=result)

    else:
        flash('Invalid model choice')
        return redirect(request.url)

    
@app.route('/yolo_result')
def yolo_result():
    try:
        file = request.files['file']
        pred_classes = predict_yolo(file)

        image_bytes = BytesIO()
        image = Image.open(file)
        image.save(image_bytes, format='PNG')
        image_base64 = 'data:image/png;base64,' + base64.b64encode(image_bytes.getvalue()).decode('utf-8')

        return render_template('yolo_result.html', pred_classes=pred_classes, result_image=image_base64)

    except Exception as e:
        return f"Error: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True)

