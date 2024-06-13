import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st

# Load the model
Model_path = "model_inceptionV3.h5"
model = load_model(Model_path)

# Function to predict image class
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)[0]

    class_labels = [
       "Burger", "Butter Naan", "Chai", "Chapati", "Chole Bhature",
       "Dal Makhani", "Dhokla", "Fried Rice", "Idli", "Jalebi",
       "Kaathi Rolls", "Kadai Paneer", "Kulfi", "Masala Dosa", "Momos",
       "Paani Puri", "Pakode", "Pav Bhaji", "Pizza", "Samosa"
    ]

    result = f"This Item is {class_labels[preds]}"
    return result

# Main function to run the Streamlit app
def main():
    st.title('Food Item Classifier')
    st.text('Upload a food image...')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        result = model_predict(file_path, model)
        st.success(result)

if __name__ == '__main__':
    main()
