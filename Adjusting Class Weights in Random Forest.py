# Train a Random Forest Classifier with class weights
rf_model_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model_weighted.fit(X_train, y_train)

# Make predictions on the test data
y_pred_weighted = rf_model_weighted.predict(X_test)

# Evaluate the model
print("Accuracy Score with Class Weighting:", accuracy_score(y_test, y_pred_weighted))
print("Classification Report with Class Weighting:\n", classification_report(y_test, y_pred_weighted))
