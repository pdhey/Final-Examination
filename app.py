import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
import shutil

import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module


st.title("""
Cat 🐱 Or Dog 🐶?
	""")

img_file_buffer = st.file_uploader("Upload an image: ")

try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview 👀 Of Given Image!
		""")
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )

	st.write("""
		**Just Click The Button! 😄**
		""")
except:
	st.write("""
		### ❗ Any Picture hasn't selected yet!!!
		""")

st.text("""""")
submit = st.button("DOG or CAT")

def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path, 
            target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array/255.0
    img_array = img_array.reshape((1, 50, 50, 1))   
    prediction =loaded_model.predict(img_array)    
    return prediction

def generate_result(prediction):
	st.write("""
	## RESULT
		""")
	if prediction[0]<0.5:
	    st.write("""
	    	## It's a CAT 🐱!!!
	    	""")
	else:
	    st.write("""
	    	## It's a DOG 🐶!!!
	    	""")

if submit:
	try:
		# save image on that directory
		save_img("test_image.png", img_array)

		image_path = "test_image.png"
		# Predicting
		st.write("👁Loading...")

		model_path_h5 = "model.h5"
		model_path_json = "model.json"
		json_file = open(model_path_json, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(model_path_h5)

		loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')

		prediction = processing(image_path)

		generate_result(prediction)

	except:
		st.write("""
		### ❗ Oops... Something Is Going Wrong
			""")
