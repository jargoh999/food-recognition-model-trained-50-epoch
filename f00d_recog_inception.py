import streamlit as ss


import numpy as np #standard
import plotly.express as px  #plots and graphing lib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def dic_maker(arr):
  """ dis takes in arr [[prob(1),prob(2),prob(3)......prob(n)]]
   and outputs [(1,prob(1)),(2,prob(2))]
   (basically some formatting to make life easier)"""
  dict_ = {}
  for ind in range(len(arr[0])):
    dict_[ind] = arr[0][ind]
  return sorted(dict_.items(), key=lambda x: x[1],reverse=True)[:3]


def dic_maker_tuple(tuple_arr):
  """ takes in [(x,y),(a,b)]
      outputs {x:y,a:b} (basically some formatting to make life easier)
  """
  dict_ = {}
  for tuple_ in tuple_arr:
    dict_[target_dict[tuple_[0]]] = tuple_[1]
  return dict_


def inception_no_gen(image):
  """ 
  prediction happens in this function
  super important, takes in image_path (/content/test_1/test/111.jpg)
  outputs: {1:prob(1),2:prob(2)}
  """
  #image_1 = tensorflow.keras.preprocessing.image.load_img(image_path)


  input_arr = tensorflow.keras.preprocessing.image.img_to_array(image)
  input_arr = preprocess_input(input_arr)
  input_arr = tensorflow.image.resize(input_arr,size = (256,256))
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model_saved.predict(input_arr)
  return dic_maker_tuple(dic_maker(predictions))

def plot_pred_final(test_imgs):
  """
  dis takes in {1:prob(1),2:prob(2)}
  and plots a SUPER NORMIE PLOT to make it easier for SRM FACULTY(or they might flip out like the bunch of idiots they are)
  """
  #test_imgs = glob(image_path_custom + '/*/*.jpeg')
  fig = make_subplots(rows = 2, cols = 2)
  pred_list = inception_no_gen(test_imgs)
  fig.append_trace(go.Image(z = np.array(test_imgs)),1,1)
  fig.append_trace(go.Bar(y = list(pred_list.keys()), x = list(pred_list.values()),orientation = 'h'),1,2)
  fig.update_layout(width = 1750, height = 800,title_text = "Custom Predictions",showlegend = False)
  return fig

#------streamlit starts here----------------

model_saved = tensorflow.keras.models.load_model("inception_food_rec_50epochs.h5")
target_dict = {0:"Bread",1:"Dairy_product",2:"Dessert",3:"Egg",4:"Fried_food",
                 5:"Meat",6:"Noodles/Pasta",7:"Rice",8:"Seafood",9:"Soup",10:"veggies/Fruit"}
ss.set_page_config(
    page_title="Food Recognition App",
    page_icon="🍽️",
    layout="wide"
)

ss.title("Food Recognition App")
ss.markdown("""
## Welcome to the Food Recognition System

Upload an image of your food, and our AI will identify what type of food it is!

This app uses a state-of-the-art InceptionV3 model trained on 16,600 food images.
""")

# Create two columns for better layout
left_column, right_column = ss.beta_columns([1, 2])

with left_column:
    ss.markdown("""
    ### How it works:
    1. Upload an image of your food
    2. Get instant recognition results
    3. View the confidence scores
    
    ### Supported Food Categories:
    - Bread
    - Dairy Products
    - Desserts
    - Eggs
    - Fried Foods
    - Meat
    - Noodles/Pasta
    - Rice
    - Seafood
    - Soups
    - Vegetables/Fruits
    
    ### Technical Details:
    - Model: InceptionV3
    - Training Data: 16,600 food images
    - Training Epochs: 50
    - Training Accuracy: 90%
    - Validation Accuracy: 76%
    
    Dataset sources:
1. [Kaggle Food11](https://www.kaggle.com/trolukovich/food11-image-dataset) - 16,600 food images across 11 categories
2. [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) - 101 food categories with 101,000 images
3. [Food-5K](https://food-5k.github.io/) - 5,000 food images with fine-grained annotations
4. [FoodX-251](https://github.com/AlpacaDB/FoodX-251) - 251 food categories with 1.3M images
5. [Food-251](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-251/) - 251 food categories with 251,000 images

Note: The current model is trained on Food11 dataset, but can be extended with additional datasets for improved accuracy.
    """)

with right_column:
    # Create file uploader
    uploaded_file = ss.file_uploader("Upload an image of your food", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        ss.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process and predict
        if ss.button('Predict Food Type', key='predict_button'):
            predictions = inception_no_gen(image)
            
            # Display predictions using Streamlit's default styling
            ss.markdown("### Prediction Results")
            for category, probability in predictions.items():
                ss.markdown(f"- {category}: {probability:.2%}")

ss.markdown("""
---
## About the Model

This app uses InceptionV3, a powerful deep learning model that:
- Achieves 90% accuracy on training data
- Has 76% accuracy on validation data
- Was trained for 50 epochs
- Uses advanced techniques like label smoothing and batch normalization

The model was trained on a diverse dataset of 16,600 food images.
""")

if uploaded_file:
    image = Image.open(uploaded_file)
    predictions = inception_no_gen(image)
    
    # Create a more visually appealing prediction display
    ss.markdown("""
    <div class="prediction-box">
        <div class="prediction-title">Prediction Results</div>
    """, unsafe_allow_html=True)
    
    for category, probability in predictions.items():
        ss.markdown(f'<div class="prediction-item">{category}: <span class="confidence-score">{probability:.2%}</span></div>', unsafe_allow_html=True)
    
    ss.markdown("</div>", unsafe_allow_html=True)
  




