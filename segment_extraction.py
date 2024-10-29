import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
pcs_data = pd.read_csv('OperationalPC.csv')
ev_cat_data = pd.read_csv('ev_cat_01-24.csv')
ev_makers_data = pd.read_csv('EV Maker by Place.csv')
vehicle_class_data = pd.read_csv('Vehicle Class - All.csv')
ev_sales_data = pd.read_csv('ev_sales_by_makers_and_cat_15-24.csv')

# Preprocess and merge relevant data
def preprocess_data():
    # Process PCS data
    pcs_data['No. of Operational PCS'] = pd.to_numeric(pcs_data['No. of Operational PCS'])
    
    # Process EV category data
    ev_cat_data['Date'] = pd.to_datetime(ev_cat_data['Date'], format='%d/%m/%y', errors='coerce')
    ev_cat_data['Year'] = ev_cat_data['Date'].dt.year
    ev_cat_yearly = ev_cat_data.groupby('Year').sum().reset_index()
    
    # Process vehicle class data
    vehicle_class_data['Total Registration'] = vehicle_class_data['Total Registration'].str.replace(',', '').astype(int)
    
    # Merge datasets
    merged_data = pd.merge(pcs_data, ev_makers_data, on='State', how='outer')
    merged_data = pd.merge(merged_data, vehicle_class_data, on='State', how='outer')
    
    # Fill NaN values with 0 for numeric columns
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    merged_data[numeric_columns] = merged_data[numeric_columns].fillna(0)
    
    return merged_data

merged_data = preprocess_data()

# Select features for clustering
features = ['No. of Operational PCS', 'Total Registration']
X = merged_data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-means clustering
def perform_kmeans(data, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_curve.png')
    plt.close()

    # Choose optimal k (this is a simple method, in practice you might want to use more sophisticated techniques)
    optimal_k = 3  # This would typically be determined by analyzing the elbow curve

    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = final_kmeans.fit_predict(data)
    return cluster_labels

cluster_labels = perform_kmeans(X_scaled)
merged_data['Cluster'] = cluster_labels

# 2. Principal Component Analysis (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Visualize PCA results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
plt.title('PCA of State Segments')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter)
plt.savefig('pca_visualization.png')
plt.close()

# 3. Decision Trees
# For this example, we'll use 'Total Registration' as the target variable
X = merged_data[['No. of Operational PCS']]
y = merged_data['Total Registration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Total Registration')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Segment extraction and analysis complete. Check the generated images for visualizations.")

# Additional analysis: Profiling the clusters
cluster_profiles = merged_data.groupby('Cluster').mean()
print("\nCluster Profiles:")
print(cluster_profiles)

# Save cluster profiles to CSV
cluster_profiles.to_csv('cluster_profiles.csv')

print("\nCluster profiles saved to 'cluster_profiles.csv'")

# Analyze EV sales trends
ev_sales_data_melted = ev_sales_data.melt(id_vars=['Cat', 'Maker'], var_name='Year', value_name='Sales')
ev_sales_data_melted['Sales'] = pd.to_numeric(ev_sales_data_melted['Sales'], errors='coerce')

# Plot top 5 EV makers by total sales
top_5_makers = ev_sales_data_melted.groupby('Maker')['Sales'].sum().nlargest(5)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_5_makers.index, y=top_5_makers.values)
plt.title('Top 5 EV Makers by Total Sales (2015-2024)')
plt.xlabel('Maker')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_5_ev_makers.png')
plt.close()

print("Additional EV sales analysis complete. Check 'top_5_ev_makers.png' for visualization.")