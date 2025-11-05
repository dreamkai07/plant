import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64

# --- Page Config ---
st.set_page_config(page_title="FarmSense | Plant Disease Classifier", layout="wide")

# --- Background Setup ---
working_dir = os.path.dirname(os.path.abspath(__file__))
background_path = os.path.join(working_dir, "background.jpg")

if os.path.exists(background_path):
    with open(background_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5, h6, p, div {{
            font-family: 'Segoe UI', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Load Model and Class Info ---
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# --- Load Disease Info File ---
disease_info_path = f"{working_dir}/disease_info.json"
if os.path.exists(disease_info_path):
    with open(disease_info_path, "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}

# --- Helper Functions ---
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# --- Streamlit App ---
st.markdown(
    """
    <h1 style='text-align: center; color: #1B5E20;'>
        🌾 <b>FarmSense</b> | Plant Disease Classifier 🌿
    </h1>
    <p style='text-align: center; color: white; font-size: 18px;'>
        Upload a plant leaf image to detect diseases and get smart farming advice.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 1])

    with col1:
        resized_img = image.resize((400, 400))
        st.image(resized_img, caption="Uploaded Image", width=400)

    with col2:
        if st.button('🔍 Classify'):
            # Predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'🌿 Prediction: {prediction}')

            # Display detailed info if available
            if prediction in disease_info:
                info = disease_info[prediction]

                st.markdown("### 🦠 Cause")
                st.markdown(
                    f"<p style='background-color:rgba(255,255,255,0.7);padding:10px;border-radius:10px;'>{info.get('cause','Not available')}</p>",
                    unsafe_allow_html=True
                )

                st.markdown("### 🍃 Symptoms")
                st.markdown(
                    f"<p style='background-color:rgba(255,255,255,0.7);padding:10px;border-radius:10px;'>{info.get('symptoms','Not available')}</p>",
                    unsafe_allow_html=True
                )

                st.markdown("### 💊 Solution")
                st.markdown(
                    f"<p style='background-color:rgba(255,255,255,0.7);padding:10px;border-radius:10px;'>{info.get('solution','Not available')}</p>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No additional information available for this disease.")
