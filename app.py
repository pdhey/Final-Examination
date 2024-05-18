import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import random
import webbrowser

st.title("Cat 🐱 Or Dog 🐶?")

img_file_buffer = st.file_uploader("Upload an image: ")

if img_file_buffer is not None:
    try:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        st.write("Preview 👀 Of Given Image!")
        st.image(image, use_column_width=True)
        st.write("Just Click The Button! 😄")
    except Exception as e:
        st.write(f"### ❗ Error: {e}")
else:
    st.write("### ❗ No picture has been selected yet!!!")

submit = st.button("DOG or CAT")

def processing(testing_image_path, model):
    IMG_SIZE = 50
    img = load_img(testing_image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 1))
    prediction = model.predict(img_array)
    return prediction

def generate_result(prediction):
    st.write("## RESULT")
    if prediction[0] < 0.5:
        st.write("## It's a CAT 🐱!!!")
    else:
        st.write("## It's a DOG 🐶!!!")

if submit and img_file_buffer is not None:
    try:
        # Save the uploaded image
        save_path = "test_image.png"
        image.save(save_path)

        # Load the model
        model_path_h5 = "model.h5"
        model_path_json = "model.json"
        with open(model_path_json, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_path_h5)
        loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Process the image and predict
        prediction = processing(save_path, loaded_model)
        generate_result(prediction)
    except Exception as e:
        st.write(f"### ❗ Oops... Something went wrong: {e}")
else:
    if submit:
        st.write("### ❗ No picture has been selected yet!!!")
