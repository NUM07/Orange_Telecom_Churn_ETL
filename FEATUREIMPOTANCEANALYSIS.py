import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Display the first few rows to confirm data is loaded
print(df.head())

# Step 2: Preprocessing
# Convert 'Churn' to numerical (if it's a boolean or categorical column)
df['Churn'] = df['Churn'].map({True: 1, False: 0})

# Step 3: Split the data into features (X) and target (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target (Churn)

# Handle categorical columns (convert them to numerical using one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Step 4: Train a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier and fit it to the training data
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=False)
rf_model.fit(X_train, y_train)

# Step 5: Feature Importance Analysis
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# Plot feature importances
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), colormap='viridis')

# Add title and labels
plt.title("Feature Importance in Churn Prediction")
plt.xlabel("Features")
plt.ylabel("Importance Score")

# Rotate x-axis labels for better visibility and adjust layout
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align them to the right
plt.tight_layout()  # Automatically adjust layout to ensure labels fit within the figure

# Display the plot
plt.show()
