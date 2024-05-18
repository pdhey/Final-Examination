import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
@st.cache_data
def load_model():
    model = tf.keras.models.load_model('cats_and_dogs_small_1.h5')
    return model

model = load_model()

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img_reshape = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

# Main Streamlit app code
st.write("""# Dog vs Cat Classification""")
file = st.file_uploader("Choose a photo from computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Error handling for prediction
    try:
        prediction = import_and_predict(image, model)
        class_names = ['Dog', 'Cat']
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        st.success(string)
    except Exception as e:
        st.error("Error making prediction: {}".format(e))
