import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import gdown


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

if not os.path.exists(model_path):
    os.makedirs(f"{working_dir}/trained_model", exist_ok=True)
    gdown.download("https://drive.google.com/uc?id=1j7utM1kojCRoec0fhb9uYyUVAlB8rFTu", model_path, quiet=False)
    
    # Fix potential quantization_config issue in newer TensorFlow versions
    import h5py
    try:
        with h5py.File(model_path, 'r+') as f:
            model_config = f.attrs.get('model_config')
            if model_config is not None:
                config_str = model_config.decode('utf-8') if isinstance(model_config, bytes) else model_config
                config_dict = json.loads(config_str)
                
                modified = [False]
                def remove_quant_config(config):
                    if isinstance(config, dict):
                        if 'quantization_config' in config:
                            del config['quantization_config']
                            modified[0] = True
                        for k, v in list(config.items()):
                            remove_quant_config(v)
                    elif isinstance(config, list):
                        for item in config:
                            remove_quant_config(item)

                remove_quant_config(config_dict)
                if modified[0]:
                    new_config_str = json.dumps(config_dict)
                    if isinstance(model_config, bytes):
                        new_config_str = new_config_str.encode('utf-8')
                    f.attrs['model_config'] = new_config_str
    except Exception as e:
        print(f"Failed to patch h5 file: {e}")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')