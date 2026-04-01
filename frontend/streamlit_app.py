import streamlit as st
import requests

st.title("📈 Lag-Llama Forecasting UI")

api_url = st.text_input("API URL", "https://your-api.onrender.com/forecast")

uploaded_file = st.file_uploader("Upload CSV (date, value)", type=["csv"])

prediction_length = st.number_input("Prediction Length", value=28)

context_lengths = st.text_input("Context Lengths (comma separated)", "30,60,90")

if st.button("Run Forecast"):
    if uploaded_file is not None:
        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            api_url,
            files={"file": uploaded_file},
            data={
                "prediction_length": prediction_length,
                "context_lengths": context_lengths
            }
        )

        if response.status_code == 200:
            data = response.json()

            st.success("Forecast Generated!")

            for ctx, values in data["contexts"].items():
                st.subheader(f"Context {ctx}")
                st.line_chart(values)
        else:
            st.error("API Error")