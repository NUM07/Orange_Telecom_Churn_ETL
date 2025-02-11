import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Select only numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns only
correlation_matrix = df_numeric.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))  # Increase the size to accommodate the labels
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add a title to the heatmap
plt.title('Correlation Heatmap of Features', fontsize=16)

# Adjust layout to ensure labels and headers are fully visible
plt.tight_layout(pad=3)  # Adjust the padding further if needed

# Show the plot
plt.show()
