import streamlit as st
import requests
import pandas as pd
from PIL import Image

st.set_page_config(page_title="AI Medical Platform", layout="wide")
st.title("🩺 Advanced AI Medical Intelligence Platform")
st.warning("AI-assisted output only. Not a medical diagnosis.")

api = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
page = st.sidebar.radio("Menu", ["Predict", "History", "About"])

if page == "Predict":
    uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=350)

        if st.button("Analyze Image"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            r = requests.post(f"{api}/predict", files=files, timeout=180)
            if r.status_code != 200:
                st.error(r.text)
            else:
                data = r.json()
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Prediction")
                    st.success(data["prediction"])
                    st.metric("Confidence", f"{data['confidence']*100:.2f}%")
                    st.metric("Latency", f"{data['latency_ms']:.2f} ms")
                with c2:
                    st.subheader("Grad-CAM")
                    st.image(f"{api}{data['gradcam']}", use_column_width=True)

                st.subheader("AI Medical Report")
                st.write(data["report"])

elif page == "History":
    st.subheader("Prediction History")
    limit = st.slider("Rows", 5, 100, 20, 5)
    if st.button("Load History"):
        r = requests.get(f"{api}/history?limit={limit}", timeout=60)
        if r.status_code == 200:
            rows = r.json()
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.error(r.text)

else:
    st.markdown("""
### About
- DenseNet121 chest X-ray classification
- Grad-CAM explainability
- Gemini AI report generation
- FastAPI + SQLite + Streamlit
""")
