# app.py
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from vit_model import vit_predict  # Implement vit_predict using a pre-trained ViT model
from yolo_model import yolo_predict  # Implement yolo_predict using a pre-trained YOLO model

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        photo_path = f"uploads/{photo.filename}"
        photo.save(photo_path)

        model_choice = request.form.get('model_choice')

        if model_choice == 'vit':
            result = vit_predict(photo_path)  # Implement vit_predict function
            return f'ViT Prediction: {result}'
        elif model_choice == 'yolo':
            result_path = yolo_predict(photo_path)  # Implement yolo_predict function
            return render_template('yolo_result.html', result_path=result_path)

    return 'Error uploading image.'

if __name__ == '__main__':
    app.run(debug=True)
