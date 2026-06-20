
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Brain Tumor Detection using Deep Learning")

st.markdown("""
Upload a **Brain MRI image** and the trained CNN model will predict the tumor type.

### Supported Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor
""")

st.divider()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Tumor", use_container_width=True):

        with st.spinner("Analyzing MRI Scan..."):

            # Preprocess Image
            img = image.resize((128, 128))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            prediction = model.predict(img, verbose=0)[0]

            predicted_index = np.argmax(prediction)

            predicted_class = class_names[predicted_index]

            confidence = prediction[predicted_index] * 100

        st.divider()

        st.subheader("🧠 Prediction Result")

        if predicted_class == "No Tumor":

            st.success(
                f"✅ {predicted_class}\n\nConfidence : {confidence:.2f}%"
            )

        elif predicted_class == "Glioma":

            st.error(
                f"🔴 {predicted_class}\n\nConfidence : {confidence:.2f}%"
            )

        elif predicted_class == "Meningioma":

            st.warning(
                f"🟠 {predicted_class}\n\nConfidence : {confidence:.2f}%"
            )

        else:

            st.info(
                f"🔵 {predicted_class}\n\nConfidence : {confidence:.2f}%"
            )

        st.divider()

        st.subheader("📊 Prediction Probabilities")

        for cls, prob in zip(class_names, prediction):

            st.write(f"**{cls}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")

        st.divider()

        st.metric(
            label="Highest Confidence",
            value=f"{confidence:.2f}%"
        )

st.divider()

st.caption(
    "Brain Tumor Classification using Convolutional Neural Networks (CNN)"
)

