import streamlit as st
import pandas as pd
import plotly.express as px

# ================= PAGE =================
st.set_page_config(page_title="RFM Analysis", layout="wide")

st.title("📊 RFM Analysis Dashboard")

# ================= LOAD DATA =================
df = pd.read_csv("Online_Retail_Small.csv", encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceNo"] = df["InvoiceNo"].astype(str)

# ================= RFM TABLE =================
rfm_table = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Quantity": "sum"
}).reset_index()

rfm_table.columns = ["CustomerID", "Frequency", "Monetary"]

# ================= KPI CARDS =================
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Customers", rfm_table["CustomerID"].nunique())
col2.metric("🛒 Avg Frequency", round(rfm_table["Frequency"].mean(), 2))
col3.metric("💰 Avg Monetary", round(rfm_table["Monetary"].mean(), 2))

st.markdown("---")

# ================= CHART FUNCTION (BORDER CARD) =================
def chart_box(fig, title):
    st.subheader(title)

    st.markdown("""
    <div style="
        border:2px solid #4CAF50;
        border-radius:15px;
        padding:12px;
        box-shadow:2px 2px 10px rgba(0,0,0,0.08);
        background-color:white;
        margin-bottom:20px;
    ">
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ================= 1. RFM SCATTER =================
fig1 = px.scatter(
    rfm_table,
    x="Frequency",
    y="Monetary",
    color="Monetary",
    title="RFM Distribution"
)

chart_box(fig1, "📊 RFM Distribution")

# ================= 2. TOP CUSTOMERS =================
top_customers = rfm_table.sort_values("Monetary", ascending=False).head(10)

fig2 = px.bar(
    top_customers,
    x="CustomerID",
    y="Monetary",
    title="Top 10 Customers"
)

chart_box(fig2, "🏆 Top Customers")

# ================= 3. FREQUENCY DISTRIBUTION =================
fig3 = px.histogram(
    rfm_table,
    x="Frequency",
    nbins=20,
    title="Frequency Distribution"
)

chart_box(fig3, "📦 Frequency Distribution")

# ================= 4. SEGMENTS =================
def segment(x):
    if x > 50000:
        return "High Value"
    elif x > 20000:
        return "Medium"
    else:
        return "Low"

rfm_table["Segment"] = rfm_table["Monetary"].apply(segment)

fig4 = px.pie(
    rfm_table,
    names="Segment",
    title="Customer Segments"
)

chart_box(fig4, "🎯 Customer Segments")
