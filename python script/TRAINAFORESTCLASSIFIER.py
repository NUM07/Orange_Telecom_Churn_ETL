
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv("C:/Users/ASUS/Desktop/orange_telecom_project/Raw_Data/churn-bigml-80.csv")
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Ensure the data has no null values
if data.isnull().any().any():
    print("Error: The dataset contains missing values. Please clean the data and try again.")
    exit()

# Drop non-numeric columns if present (keep only numeric features for clustering)
numeric_data = data.select_dtypes(include=[np.number])

# Preprocess the data (standardize the features)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply K-means clustering
num_clusters = 4  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Apply PCA to reduce data to 2D for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Get the centroids of the clusters
centroids = kmeans.cluster_centers_

# Apply PCA transformation to the centroids
centroids_pca = pca.transform(centroids)

# Plot the clusters with centroids
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['Cluster'], cmap='viridis', label='Data Points')

# Plot the centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')

# Customize the plot
plt.title('K-means Clustering with PCA (2D Visualization with Centroids)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

# Save the data with cluster labels to a new CSV
output_file_path = 'C:\\Users\\ASUS\\Desktop\\orange_telecom_project\\Processed_Data\\data_with_clusters.csv'
data.to_csv(output_file_path, index=False)
print(f"Clustered data saved to {output_file_path}")
