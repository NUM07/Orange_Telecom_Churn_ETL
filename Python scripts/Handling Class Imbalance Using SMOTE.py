import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignore warnings to make the output clean
warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Display the first few rows of the dataset
print("Dataset head:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Convert categorical columns to numeric using Label Encoding
label_encoder = LabelEncoder()
df['State'] = label_encoder.fit_transform(df['State'])
df['International plan'] = label_encoder.fit_transform(df['International plan'])
df['Voice mail plan'] = label_encoder.fit_transform(df['Voice mail plan'])

# Ensure 'Churn' column is a boolean (if it's not already)
df['Churn'] = df['Churn'].astype(bool)

# Split the data into features (X) and target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check the shape of the resampled dataset
print("\nAfter SMOTE, X_train_res shape:", X_train_res.shape)
print("After SMOTE, y_train_res shape:", y_train_res.shape)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with n_jobs=1 to avoid parallelization issues
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)

# Fit the model using the training data with SMOTE applied
grid_search.fit(X_train_res, y_train_res)

# Get the best estimator from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show the feature importances from the best model
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))
