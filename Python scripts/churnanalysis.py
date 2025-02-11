import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Ensure 'Churn' column is numeric (0 for False, 1 for True)
df['Churn'] = df['Churn'].astype(int)

# Verify conversion
print(df['Churn'].value_counts())  # Should output 2278 for 0 and 388 for 1
print(df.dtypes)  # Check if Churn is now an integer

# Define the features to analyze
features = ['Account length', 'Total day minutes', 'Total day charge', 
            'Total eve minutes', 'Total eve charge', 'Total night minutes', 
            'Total night charge', 'Total intl minutes', 'Total intl charge']

# Set up the figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(14, 12))  # 3 rows, 3 columns
axes = axes.flatten()  # Flatten the array to easily loop over

# Plot histograms for each feature
for i, feature in enumerate(features):
    sns.histplot(data=df, x=feature, hue='Churn', kde=True, multiple='stack', 
                 palette='Set1', ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} by Churn', fontsize=12)
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].grid(True)
    axes[i].tick_params(axis='both', which='major', labelsize=8)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)
plt.show()
