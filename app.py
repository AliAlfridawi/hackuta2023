from io import BytesIO

from flask import Flask, render_template, request, send_file, jsonify, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import json
from roboflow import Roboflow
import os


app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

    # cache defintion 
@app.after_request 
def add_header(response):    
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"     
    response.headers["Cache-Control"] = "public, max-age=0"     
    return response


@app.route("/") 
def hello(): 
    return render_template('main.html') 

@app.route('/functional')
def functional():
    return render_template('functional.html')

@app.route("/main")  
def main(): 
    return render_template('main.html')

@app.route("/moreabout")  
def moreabout(): 
    return render_template('moreabout.html')

@app.route("/contact")  
def contact(): 
    return render_template('contact.html')




# Specify the upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
  
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        #delete old files
        dir_path = UPLOAD_FOLDER
        try:
            os.rmdir(dir_path)
        except OSError: pass
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        image = request.files['image']
        # Process the image as needed
        # Save the image, perform analysis, etc.
        print(image)

        # Save the image to the upload folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
        saved_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)


        # Process the image as needed
        # For example, you can get the saved file path
        print(saved_image_path)

        rf = Roboflow(api_key="zHtICeX8iyzeTtCNzHgI")
        project = rf.workspace().project("water-damage-finder")
        model = project.version(2).model

        photoName = saved_image_path
        # infer on a local image
        predictions = model.predict(photoName, confidence=40, overlap=30).json()
        print(predictions)

        # visualize your prediction
        annotated_image_name = "prediction.jpg"
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_image_name)

        model.predict(photoName, confidence=40, overlap=30).save(annotated_image_path)  

        predictions.update({"annotated_image_path": annotated_image_path})

        return jsonify(predictions), 200
    else:
        return 'No image provided', 400


if __name__ == "__main__":
    app.run(host='localhost', debug=True, port=5001)