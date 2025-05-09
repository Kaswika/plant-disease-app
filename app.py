import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import PIL.Image

# Load your trained model
model = load_model("plant_disease_model.h5")

# List of class names (update based on your model)
classes = ['Tomato - Healthy', 'Tomato - Late Blight', 'Potato - Early Blight', 'Potato - Healthy']

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the crop disease.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = PIL.Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img_to_array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    # Show results
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence}%")

