import pandas as pd

# Read original CSV
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# Keep first 10000 rows
small_df = df.head(10000)

# Save new CSV
small_df.to_csv("Online_Retail_Small.csv", index=False)

print("Done! New file created: Online_Retail_Small.csv")