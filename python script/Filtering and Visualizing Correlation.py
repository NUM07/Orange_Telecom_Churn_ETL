import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('telecom_churn.csv')

# Check for missing data
print(df.isnull().sum())

# Handle categorical data (e.g., converting to numeric)
label_encoder = LabelEncoder()

# Convert categorical columns to numerical values (e.g., 'International plan', 'Voice mail plan')
df['International plan'] = label_encoder.fit_transform(df['International plan'])
df['Voice mail plan'] = label_encoder.fit_transform(df['Voice mail plan'])
df['State'] = label_encoder.fit_transform(df['State'])

# Handle missing values if any
df.fillna(df.mean(), inplace=True)  # Replace missing numerical values with the mean

# Separate the features (X) and target (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable (Churn)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training and Test data split completed.")
