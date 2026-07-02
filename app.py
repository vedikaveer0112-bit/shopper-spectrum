import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ================= PAGE CONFIG (ONLY ONCE) =================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#2E8B57;'>
🛒 Shopper Spectrum
</h1>

<h3 style='text-align:center; color:gray;'>
Customer Segmentation & Product Recommendation System
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png",
    width=80
)

st.sidebar.title("Shopper Spectrum")
st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📌 Navigation
- Home (Dashboard)
- Customer Segmentation (RFM)
- Product Recommendation
""")

st.sidebar.markdown("---")
st.sidebar.success("📊 Dashboard Active")

# ================= LOAD MODELS =================
try:
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ================= LOAD DATA =================
try:
    df = pd.read_csv("Online_Retail_Small.csv", encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# ================= OVERVIEW =================
st.subheader("📌 Project Overview")

st.write("""
Shopper Spectrum helps businesses:
- 👤 Customer Segmentation (K-Means)
- 🎯 Product Insights
- 📊 Sales Analysis
""")

st.markdown("---")

# ================= METRICS =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("🛍 Products", df["Description"].nunique())
col2.metric("👥 Customers", df["CustomerID"].nunique())
col3.metric("🌍 Countries", df["Country"].nunique())
col4.metric("🧾 Orders", df["InvoiceNo"].nunique())

st.markdown("---")

# ================= TOP PRODUCTS =================
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="Quantity",
    y="Description",
    orientation="h",
    title="Top 10 Selling Products"
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ================= CUSTOMER SEGMENTATION =================
st.subheader("👤 Customer Segmentation (RFM Demo)")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input("Recency", 0, 1000, 30)

with col2:
    frequency = st.number_input("Frequency", 0, 1000, 5)

with col3:
    monetary = st.number_input("Monetary", 0, 100000, 1000)

if st.button("Predict Segment"):

    data = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]

    segments = {
        0: ("🟢 Regular Customer", "Low engagement customers"),
        1: ("🔵 Loyal Customer", "Repeat buyers"),
        2: ("🟡 High Value Customer", "Premium spending customers"),
        3: ("🔴 At Risk Customer", "Need re-engagement")
    }

    name, desc = segments.get(cluster, ("Unknown", "No info"))

    st.success(f"Segment: {name}")
    st.info(desc)

st.markdown("---")

# ================= FOOTER =================
st.markdown("""
<div style='text-align:center;color:gray;'>
<h3>🛒 Shopper Spectrum</h3>
Machine Learning Project using Streamlit, Scikit-Learn & Plotly
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    "<h4 style='text-align:center; color:#2E8B57;'>👩‍💻 Developed by Vedika Veer</h4>",
    unsafe_allow_html=True
)