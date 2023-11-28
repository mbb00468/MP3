from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from flask import Flask, render_template, request
import os
from flask import flash, redirect
from ultralytics import YOLO
from PIL import Image
import warnings
import io
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the ViT model and processor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def predict_vit(image):
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
    
    

yolo_model = YOLO('yolov8n.pt')



def predict_yolo(image):
    try:
        results = model(image, verbose=False)  # results list

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the POST request (e.g., process the uploaded file)
        return render_template('index.html')
    else:
        # Render the HTML form for GET requests
        return '''
        <!doctype html>
        <title>Upload an Image</title>
        <h1>Upload an Image</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''



@app.route('/upload', methods=['POST'])
def upload_file():
    # ... (previous code)

    model_choice = request.form.get('model_choice')

    if model_choice == 'vit':
        uploaded_file = request.files['file']
        if uploaded_file:
            # Convert the uploaded file to a PIL Image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            
            # Process the image and get the prediction result
            result = predict_vit(image)
            return render_template('index.html', result=result)

    elif model_choice == 'yolo':
        # Call function for YOLO model prediction
        # Replace the following line with the actual code for YOLO prediction
        yolo_prediction_result = 'YOLO model prediction result: [Replace with the result]'
        return render_template('yolo_result.html', result=yolo_prediction_result)

    else:
        flash('Invalid model choice')
        return redirect(request.url)

@app.route('/yolo_result/<filename>')
def yolo_result(filename):
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('yolo_result.html', result_path=result_path)

if __name__ == '__main__':
    app.run(debug=True)

