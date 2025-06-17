import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config FIRST
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")

# Load your trained model
@st.cache_resource
def load_pneumonia_model():
    return load_model('pneumonia_model.h5')

model = load_pneumonia_model()

# App title
st.title("🫁 Pneumonia Detection App")
st.markdown("Upload a chest X-ray image, and the model will predict if it's **Pneumonia** or **Normal**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        label = "Pneumonia" if prediction > 0.5 else "Normal"

        # Display result
        st.subheader(f"🔎 Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2f}%")

        # Optional: Plot image with result
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{label} ({confidence:.2f}%)")
        st.pyplot(fig)
