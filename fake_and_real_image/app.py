import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


#load the model
def load_model():
    model = tf.keras.models.load_model('fake_img_model.h5')
    return model
model = load_model()
class_names = ['FAKE', 'REAL']
#preprocess the image
def preprocess_image(image):
    image = Image.open(image).convert('L')  
    image = image.resize((150, 150))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)   
    img_array = np.expand_dims(img_array, axis=-1)  
    return img_array

#streamlit ui app
st.title('Fake Image Detection')
st.write('Upload an image to detect if it is a fake image')
image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if st.button('Predict'):
    if image is not None:
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        label = class_names[int(prediction[0][0] >= 0.5)]
        confidence = float(prediction[0][0]) if label == 'REAL' else 1 - float(prediction[0][0])
        st.image(image, use_column_width=True)
        st.markdown(f"### Result: {label}  \n(Confidence: {confidence:.2%})")
    else:
        st.write('Please upload an image')  
 