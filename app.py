# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

from bike_pipeline import BikeDefectPipeline
from inference_draem import DRAEMInference

# Load environment variables
load_dotenv()

# -------------------------------
# CONFIG (from .env file)
# -------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "checkpoints/stage1/best_model.pth")
DRAEM_PATH = os.getenv("DRAEM_PATH", "checkpoints/stage2/best_model.pth")
DEVICE = os.getenv("DEVICE", "cuda")

# -------------------------------
# LOAD PIPELINES
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_stage1_pipeline():
    """Load Stage 1: Binary Classifier"""
    pipeline = BikeDefectPipeline(classifier_path=CLASSIFIER_PATH, device=DEVICE)
    return pipeline

@st.cache_resource(show_spinner=False)
def load_stage2_pipeline():
    """Load Stage 2: DRAEM Localization"""
    draem = DRAEMInference(model_path=DRAEM_PATH, device=DEVICE)
    return draem

# Load both pipelines
stage1_pipeline = load_stage1_pipeline()
stage2_pipeline = load_stage2_pipeline()

# -------------------------------
# STREAMLIT INTERFACE
# -------------------------------
st.title("Bike Damage Detection")
st.markdown("**Two-Stage Pipeline:** Binary Classification + Defect Localization")

uploaded_file = st.file_uploader("Upload a bike image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_name = uploaded_file.name

    # Save temporarily for pipeline
    temp_path = Path("temp") / img_name
    temp_path.parent.mkdir(exist_ok=True)
    img.save(temp_path)

    st.write(f"Uploaded image: `{img_name}`")

    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

    # -------------------------------
    # STAGE 1: Binary Classification
    # -------------------------------
    st.info("**Stage 1:** Running binary classifier...")

    result = stage1_pipeline.predict(str(temp_path), visualize=False, save_dir="outputs")
    is_damaged = result["is_damaged"]
    confidence = result["confidence"]

    if is_damaged:
        st.error(f"**Prediction: DAMAGED** (Confidence: {confidence:.2%})")

        # -------------------------------
        # STAGE 2: DRAEM Localization
        # -------------------------------
        st.info("**Stage 2:** Running DRAEM defect localization...")

        # Run DRAEM inference
        draem_result = stage2_pipeline.predict(
            img_np,
            use_bike_mask=True,
            refine_heatmap_flag=True,
            threshold=0.6
        )

        # Display results
        with col2:
            st.image(draem_result["heatmap_overlay"], caption="Defect Localization", use_column_width=True)

        # Show detailed results
        st.markdown("---")
        st.subheader("Detection Results")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Anomaly Score", f"{draem_result['anomaly_score']:.2%}")
        with col4:
            st.metric("Confidence", f"{draem_result['confidence']:.2%}")

        # Show heatmap
        st.image(draem_result["heatmap"], caption="Anomaly Heatmap", use_column_width=True, clamp=True)

    else:
        st.success(f"**Prediction: INTACT** (Confidence: {confidence:.2%})")
        with col2:
            st.image(img, caption="No defects detected", use_column_width=True)

    # Show probabilities
    st.markdown("---")
    st.subheader("Classification Probabilities")
    probs = result["probabilities"]
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Intact", f"{probs['intact']:.2%}")
    with col6:
        st.metric("Damaged", f"{probs['damaged']:.2%}")
