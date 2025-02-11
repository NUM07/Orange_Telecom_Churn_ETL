import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Convert 'Churn' to numeric (1 for True, 0 for False)
df['Churn'] = df['Churn'].astype(int)

# Drop non-numeric columns
df_numeric = df.drop(columns=['State', 'International plan', 'Voice mail plan'])

# Compute correlation with Churn
correlation_with_churn = df_numeric.corr()['Churn'].sort_values(ascending=False)

# Display top correlations
print("\n🔹 Top correlations with Churn:")
print(correlation_with_churn)
