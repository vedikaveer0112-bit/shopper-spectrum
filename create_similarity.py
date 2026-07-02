import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("Online_Retail_Small.csv", encoding="ISO-8859-1")

# Keep only product descriptions
df = df[["Description"]].dropna().drop_duplicates()

# Create product list
products = df["Description"].tolist()

# Convert text to vectors
cv = CountVectorizer(stop_words="english")
feature_matrix = cv.fit_transform(products)

# Compute similarity
similarity = cosine_similarity(feature_matrix)

# Save files
joblib.dump(products, "product_list.pkl")
joblib.dump(similarity, "similarity.pkl")

print("✅ product_list.pkl created")
print("✅ similarity.pkl created")