import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_orange_churn.csv")

# List of numerical columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Calculate the number of rows and columns for subplots
num_plots = len(numerical_cols)
ncols = 3  # Three columns of plots
nrows = (num_plots // ncols) + (num_plots % ncols > 0)  # Round up for extra rows if needed

# Create a figure with subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
axes = axes.flatten()

# Plot each of the columns
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}", fontsize=10, pad=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel('Frequency', fontsize=8)
    axes[i].tick_params(axis='x', rotation=45, labelsize=7)
    axes[i].tick_params(axis='y', labelsize=7)

# Hide unused axes if any
for j in range(len(numerical_cols), len(axes)):
    axes[j].set_visible(False)

# Adjust layout to avoid overlap
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Show the plot
plt.tight_layout()
plt.show()
