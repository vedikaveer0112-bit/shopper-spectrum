import streamlit as st
import numpy as np
import joblib

kmeans = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")

st.title("👤 Customer Segmentation")

recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary")

if st.button("Predict Segment"):

    data = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]

    segments = {
        0: ("🟢 Regular Customer", "Low engagement"),
        1: ("🔵 Loyal Customer", "Frequent buyers"),
        2: ("🟡 High Value Customer", "High spenders"),
        3: ("🔴 At Risk Customer", "Needs attention")
    }

    name, desc = segments.get(cluster, ("Unknown", ""))

    st.success(name)
    st.info(desc)
