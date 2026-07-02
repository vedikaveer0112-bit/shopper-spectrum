import streamlit as st
import pandas as pd

st.title("🎯 Product Recommendation")

df = pd.read_csv("Online_Retail_Small.csv", encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

product = st.text_input("Enter Product Name")

if st.button("Recommend"):

    if product in df["Description"].values:

        similar = df[df["Description"] != product]["Description"].dropna().unique()[:5]

        st.write("### Recommended Products:")
        for p in similar:
            st.write("👉", p)

    else:
        st.error("Product not found")
