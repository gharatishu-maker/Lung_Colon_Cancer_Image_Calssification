import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🔬 Histopathology Image Analysis",
    page_icon="🔬",
    layout="centered"
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.keras"

# Class names must follow the training order
CLASS_NAMES = [
    "colon_aca",
    "colon_n",
    "lung_aca",
    "lung_n",
    "lung_scc"
]

# Simple explanations for each class
CLASS_INFO = {
    "colon_aca": {
        "status": "Cancer-associated class",
        "meaning": "Colon adenocarcinoma is a type of cancer that develops in the gland-forming cells lining the colon (large intestine)."
    },
    "colon_n": {
        "status": "Normal tissue class",
        "meaning": "This class represents normal colon tissue. The model has identified patterns associated with non-cancerous colon tissue."
    },
    "lung_aca": {
        "status": "Cancer-associated class",
        "meaning": "Lung adenocarcinoma is a type of lung cancer that develops in cells that produce substances such as mucus. It is one of the common types of lung cancer."
    },
    "lung_n": {
        "status": "Normal tissue class",
        "meaning": "This class represents normal lung tissue. The model has identified patterns associated with non-cancerous lung tissue."
    },
    "lung_scc": {
        "status": "Cancer-associated class",
        "meaning": "Lung squamous cell carcinoma is a type of lung cancer that develops in flat cells lining parts of the airways."
    }
}

# Load the trained model once and reuse it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Prepare uploaded image for prediction
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Main UI
st.title("Histopathology Image Analysis")
st.write(
    "Upload a colon or lung tissue image to explore its predicted "
    "classification using a deep learning model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a tissue image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded histopathology image",
        use_container_width=True
    )

    if st.button("Analyse Image", type="primary", use_container_width=True):

        try:
            with st.spinner("Analysing image..."):

                model = load_model()
                processed_image = preprocess_image(image)

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                predicted_index = int(np.argmax(predictions[0]))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[0][predicted_index]) * 100

                class_details = CLASS_INFO[predicted_class]

            st.divider()
            st.subheader("Analysis Result")

            if class_details["status"] == "Cancer-associated class":
                st.error("Model prediction: Cancer-associated tissue class")
            else:
                st.success("Model prediction: Normal tissue class")

            st.markdown(f"**Predicted class:** `{predicted_class}`")

            st.markdown(f"**Meaning:** {class_details['meaning']}")

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence indicates how strongly the model favours "
                "this class. It is not the probability that a patient "
                "has cancer."
            )

        except Exception as e:
            st.error(f"Unable to analyse the image: {e}")

st.divider()

# Medical disclaimer
st.caption(
    "For experimental and academic purposes only. "
    "This application is not a medical diagnostic tool. "
    "Please consult a medical professional for diagnosis."
)
