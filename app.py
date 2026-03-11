import streamlit as st
import numpy as np
from PIL import Image
from vehicle_detection import detect_vehicles

# Page settings
st.set_page_config(page_title="AI Traffic Signal System", layout="wide")

# Background color
page_bg = """
<style>
.stApp {
background-color: #d0f0c0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("🚦 AI Traffic Signal Optimization System")
st.write("Real-time traffic monitoring using AI")

# Upload image
uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    result, total = detect_vehicles(img)

    col1, col2 = st.columns([2,1])

    # Show detected image
    with col1:
        st.image(result, caption="Detected Vehicles", use_container_width=True)

    # Traffic signal decision
    with col2:

        st.subheader("Traffic Signal Decision")

        if total <= 5:
            signal = "🔴 RED"
            time = 60
        elif total <= 15:
            signal = "🟡 YELLOW"
            time = 30
        else:
            signal = "🟢 GREEN"
            time = 10

        st.markdown(f"## {signal}")
        st.metric("Total Vehicles", total)
        st.metric("Signal Timer (seconds)", time)
