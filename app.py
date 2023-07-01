from flask import Flask, render_template, request
import os

app = Flask(__name__, static_folder='static')
@app.route('/', methods=['GET', 'POST'])


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
    # Process the uploaded image here
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        return render_template('home.html', filename=filename)
    else:
        return "No image file selected!"
if __name__ == '__main__':
    app.run(port=8000, debug=True)