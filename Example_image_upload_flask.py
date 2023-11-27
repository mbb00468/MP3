from flask import Flask, request, redirect, url_for, flash
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
        # Call function for ViT model prediction
        # Replace the following line with the actual code for ViT prediction
        flash('ViT model prediction result: [Replace with the result]')
        return redirect(request.url)

    elif model_choice == 'yolo':
        # Call function for YOLO model prediction
        # Replace the following line with the actual code for YOLO prediction
        flash('YOLO model prediction result: [Replace with the result]')
        return redirect(url_for('yolo_result', filename=file.filename))

    else:
        flash('Invalid model choice')
        return redirect(request.url)

@app.route('/yolo_result/<filename>')
def yolo_result(filename):
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('yolo_result.html', result_path=result_path)

if __name__ == '__main__':
    app.run(debug=True)
