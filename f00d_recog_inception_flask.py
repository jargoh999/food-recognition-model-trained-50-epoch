from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = Flask(__name__)

# Model loading and initialization
def load_model():
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(11, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    model.load_weights('inception_food_rec_50epochs.h5')
    return model

model_saved = load_model()
target_dict = {
    0: "Bread",
    1: "Dairy_product",
    2: "Dessert",
    3: "Egg",
    4: "Fried_food",
    5: "Meat",
    6: "Noodles/Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "veggies/Fruit"
}

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Food Recognition App</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .container { max-width: 800px; margin: 0 auto; padding: 20px; }
                .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; }
                .result { margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Food Recognition App</h1>
                <div class="upload-area">
                    <form action="/predict" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept="image/*" required>
                        <button type="submit">Upload Image</button>
                    </form>
                </div>
                <div class="result" id="result"></div>
            </div>
        </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    try:
        # Process the image
        img = Image.open(file)
        img = img.resize((299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        preds = model_saved.predict(x)
        result = {}
        for i, prob in enumerate(preds[0]):
            result[i] = float(prob) * 100
        
        # Convert to HTML
        result_html = '<div class="result">'
        result_html += '<h3>Prediction Results:</h3>'
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for food_type, prob in sorted_results:
            result_html += f'<p>{target_dict[food_type]}: {prob:.2f}%</p>'
        result_html += '</div>'
        
        return result_html
    except Exception as e:
        return f'<div class="result"><p>Error processing image: {str(e)}</p></div>'

if __name__ == '__main__':
    app.run(debug=True)
