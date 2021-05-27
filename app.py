from tensorflow import keras
import os
from flask import Flask, render_template, request

model = keras.models.load_model('cartoon_classifier.h5')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html",img_ads = "static/image/send.png" )


@app.route('/predict', methods=['POST'])
def home():
    image = request.files["image"]
    save = image.save("static\image.jpg")
    load_image = keras.preprocessing.image.load_img("static\image.jpg",target_size=(200,200))
    image_array = keras.preprocessing.image.img_to_array(load_image)
    reshape_array = image_array.reshape(1,200,200,3)
    image = reshape_array/255
    result = model.predict(image)
    result = result.argmax()

    x = {0:"Chota Bheem",1:"Doraemon",2:"Mr.Bean",3:"Ninja Hattori",4:"Oogy and the cockroaches",5:"Shinchan",6:"Tom and Jerry"}    
    
    return render_template("index.html", result = x[result] , img_ads = "static\image.jpg" )

if __name__ == "__main__": 
    app.run(debug=True)    