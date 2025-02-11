import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Load the dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")

# Define numerical columns based on the dataset
numerical_cols = [
    'Account length', 'Area code', 'Number vmail messages', 'Total day minutes',
    'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls',
    'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls'
]

# Create a figure and a gridspec layout for subplots
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)  # hspace and wspace control the space between plots

# Plot histograms and KDE for each numerical column
for i, col in enumerate(numerical_cols):
    ax = fig.add_subplot(gs[i])
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.show()
