import pandas as pd
from sqlalchemy import create_engine

# 📌 1️⃣ Load Data
engine = create_engine('postgresql://postgres:root@127.0.0.1:54333/postgres')
query = "SELECT * FROM orange_churn;"
df = pd.read_sql(query, engine)

# 📌 2️⃣ Remove Duplicates
print(f"🔹 Duplicates Before: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"✅ Duplicates After: {df.duplicated().sum()}")

# 📌 3️⃣ Handle Missing Values (Drop or Fill)
print(f"🔹 Missing Values Before:\n{df.isnull().sum()}")
df.dropna(inplace=True)  # Adjust if necessary
print(f"✅ Missing Values After:\n{df.isnull().sum()}")

# 📌 4️⃣ Normalize Text Data
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

# 📌 5️⃣ Save Cleaned Data
df.to_csv("cleaned_orange_churn.csv", index=False)
print("\n✅ Data Cleaning Completed! Cleaned data saved as 'cleaned_orange_churn.csv'.")
