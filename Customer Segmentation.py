# customer_segmentation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("/Users/zach/Downloads/ifood_df.csv")  # Replace with your actual path

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Display columns to check what's available
print("\nColumn Names:")
print(df.columns.tolist())

# Drop irrelevant columns if they exist
columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
df_clean = df.drop(columns=columns_to_drop, errors='ignore')

# Drop missing values
df_clean = df_clean.dropna()

# Optional: print data types and descriptive stats
print("\nData Types:\n", df_clean.dtypes)
print("\nDescriptive Stats:\n", df_clean.describe())

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# Elbow Method to find optimal number of clusters
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/elbow_plot.png")  # Optional: save to file
plt.show()

# Choose optimal clusters (e.g., 4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to dataframe
df_clean['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)

df_clean['PCA1'] = reduced[:, 0]
df_clean['PCA2'] = reduced[:, 1]

# Plot clusters in 2D
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clean, palette='Set2', s=60)
plt.title('Customer Segments (via PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/cluster_plot.png")  # Optional: save to file
plt.show()
