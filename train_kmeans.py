import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Online_Retail_Small.csv", encoding="ISO-8859-1")

# Remove missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create RFM table
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Quantity": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save models
joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save RFM data (optional)
rfm.to_csv("rfm.csv")

print("✅ kmeans.pkl created")
print("✅ scaler.pkl created")
print("✅ rfm.csv created")