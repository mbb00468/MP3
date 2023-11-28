from flask import Flask, request, redirect, url_for, flash, render_template
import os

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
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print('File successfully uploaded and saved!')
        return render_template('model_choice.html')

    flash('Invalid file type. Please upload an image.')
    return redirect(request.url)

@app.route('/', methods=['GET', 'POST'])
def model_choice():
    try:
        if request.method == 'POST':

            # Get the file path from the session or form data
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.files['file'].filename)

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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        flash('An error occurred. Please try again.')
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
