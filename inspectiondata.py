import pandas as pd
from sqlalchemy import create_engine

# 📌 1️⃣ Connect to PostgreSQL & Load Data
engine = create_engine('postgresql://postgres:root@127.0.0.1:54333/postgres')
query = "SELECT * FROM orange_churn;"
df = pd.read_sql(query, engine)

# 📌 2️⃣ Inspect Data
print("🔹 Dataset Overview:")
print(df.head())  # Display first 5 rows

print("\n🔹 Dataset Info:")
print(df.info())  # Check column names, types & missing values

print("\n🔹 Missing Values Count:")
print(df.isnull().sum())  # Check missing values

print("\n🔹 Duplicates Count:", df.duplicated().sum())  # Check duplicate rows

print("\n🔹 Summary Statistics:")
print(df.describe())  # Get summary of numerical data
