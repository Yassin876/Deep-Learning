import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
# Configure page
st.set_page_config(
    page_title="X-Ray Detection",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'xray_classfication_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image):
    """Preprocess the uploaded image"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array  
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def main():
    # Title
    st.title("🫀 X-Ray Cardiomegaly Detection")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        return "model is not fond"
    
    # Upload
    uploaded_file = st.file_uploader("Upload X-ray image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        if st.button("Analyze", type="primary"):
            with st.spinner('Processing...'):
                try:
                    processed_img, display_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)
                    confidence = prediction[0][0]

                    result = "Cardiomegaly Detected" if confidence > 0.5 else "Normal"
                    percentage = confidence * 100 if confidence > 0.5 else (1 - confidence) * 100
                    
                    st.session_state['result'] = result
                    st.session_state['confidence'] = percentage
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state['result'] = None
    
    # Results

    if 'result' in st.session_state and st.session_state['result'] is not None:
        st.markdown("---")
        result = st.session_state['result']
        confidence = st.session_state['confidence']
        # عرض النتيجة بتصميم واضح
        if result == "Normal":
            st.success(f"✅ **{result}** — Confidence: {confidence:.2f}%")
        else:
            st.error(f"⚠️ **{result}** — Confidence: {confidence:.2f}%")

        # عرض تفاصيل إضافية
        st.markdown("### 🧠 Model Prediction Summary")
        st.progress(confidence / 100)

        if result == "Normal":
            st.info("💡 The X-ray appears **normal** with no signs of cardiomegaly.")
        else:
            st.warning("🚨 The model detected signs of **cardiomegaly**. Please consult a medical professional for confirmation.")




if __name__ == "__main__":
    main()

