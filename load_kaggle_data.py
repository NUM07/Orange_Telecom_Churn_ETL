# load_kaggle_data.py
# This script loads the Orange Telecom Churn dataset from a CSV file using pandas

import pandas as pd
import os

# Define the file path to the dataset
file_path = os.path.join(r"C:\Users\ASUS\Desktop\orange_telecom_project", "churn-bigml-80.csv")

# Load the dataset into a pandas DataFrame
churn_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(churn_data.head())
# Check for missing values
print(churn_data.isnull().sum())
# Check for duplicates
print("Number of duplicates:", churn_data.duplicated().sum())

# Drop duplicates
churn_data.drop_duplicates(inplace=True)
churn_data = pd.get_dummies(churn_data, columns=['State'], drop_first=True)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
churn_data['Churn'] = label_encoder.fit_transform(churn_data['Churn'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = ['Account length', 'Customer service calls']
# Add other numerical columns
churn_data[numerical_columns] = scaler.fit_transform(churn_data[numerical_columns])
print(churn_data['Churn'].value_counts())
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
