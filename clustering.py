import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

# Load the dataset
dataset = pd.read_csv('dataset.csv')
dataset.columns = dataset.columns.str.strip()  # Clean column names by removing spaces

# Step 1: Data Preparation
# Create label encoder for Activity Level
le = LabelEncoder()
dataset['Activity_Level_Encoded'] = le.fit_transform(dataset['Activity Level'].astype(str))  # Ensure Activity Level is treated as string

# Select features for clustering
features = ['Activity_Level_Encoded', 'Distance Travelled (km)', 'Calories Burned', 'App Sessions']  # Include 'App Sessions' if applicable
X = dataset[features]

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Step 2: Elbow Method and Silhouette Analysis
inertia_values = []
sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia_values.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Plot Elbow Method and Silhouette Scores
plt.figure(figsize=(12, 6))

# Plot Inertia
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range(2, 11))

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), sil_scores, marker='o')
plt.title('Silhouette Scores for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))

plt.tight_layout()
plt.show()

# Step 3: Fit KMeans with optimal k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
dataset['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 4: Visualize Clusters
plt.figure(figsize=(12, 6))
metrics = ['Distance Travelled (km)', 'Calories Burned', 'App Sessions']
cluster_means = dataset.groupby('Cluster')[metrics].mean()
cluster_means.plot(kind='bar')
plt.title('Average Metrics by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# Print Cluster Profiles
print("\nCluster Profiles:")
cluster_profiles = dataset.groupby('Cluster').agg({
    'Activity Level': lambda x: x.mode().iloc[0],  # Most common activity level in the cluster
    'Distance Travelled (km)': 'mean',
    'Calories Burned': 'mean',
    'App Sessions': 'mean'
}).round(2)

print(cluster_profiles)

# Activity Level distribution in clusters
activity_dist = pd.crosstab(dataset['Cluster'], dataset['Activity Level'])
print("\nActivity Level Distribution in Clusters:")
print(activity_dist)

# Save results
dataset.to_csv('activity_clustered_dataset.csv', index=False)
print("\nClustered dataset saved to 'activity_clustered_dataset.csv'")
