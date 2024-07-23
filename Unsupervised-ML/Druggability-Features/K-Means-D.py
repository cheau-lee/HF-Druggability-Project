import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

def load_and_clean_data(file_path):
    """Load dataset, drop duplicates, and separate binary and continuous features."""
    data = pd.read_csv(file_path, sep='\t')
    data_cleaned = data.drop_duplicates()
    
    binary_columns = data_cleaned.select_dtypes(include=['int', 'bool']).columns.tolist()
    continuous_columns = data_cleaned.select_dtypes(include='float').columns.tolist()
    
    if 'pharos_index' in binary_columns:
        binary_columns.remove('pharos_index')
    
    return data_cleaned, binary_columns, continuous_columns

def impute_missing_values(data, continuous_columns):
    """Impute missing values for continuous features."""
    imputer = SimpleImputer(strategy='mean')
    data[continuous_columns] = imputer.fit_transform(data[continuous_columns])
    return data

def remove_outliers_zscore(data, continuous_columns, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((data[continuous_columns] - data[continuous_columns].mean()) / data[continuous_columns].std())
    data_out = data[(z_scores < threshold).all(axis=1)]
    return data_out

def rank_transform(data, continuous_columns):
    """Apply rank-based inverse transformation to continuous features."""
    rank_transformer = QuantileTransformer(output_distribution='normal', n_quantiles=len(data), random_state=42)
    transformed_continuous = pd.DataFrame(rank_transformer.fit_transform(data[continuous_columns]), columns=continuous_columns)
    transformed_continuous = transformed_continuous.dropna()
    return transformed_continuous

def perform_pca(features, n_components=3):
    """Perform PCA and reduce features to the specified number of components."""
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca

def determine_optimal_k(data, k_range):
    """Determine the optimal number of clusters using the elbow method, silhouette score, and DBI."""
    wcss = []
    silhouette_scores = []
    dbi_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        
        dbi = davies_bouldin_score(data, kmeans.labels_)
        dbi_scores.append(dbi)
    
    return wcss, silhouette_scores, dbi_scores

def plot_evaluation_metrics(k_range, wcss, silhouette_scores, dbi_scores):
    """Plot the evaluation metrics for different values of k."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(k_range, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')

    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.subplot(1, 3, 3)
    plt.plot(k_range, dbi_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('DBI for Optimal k')

    plt.tight_layout()
    plt.show()

def plot_clusters_2d(features_pca, clusters, title):
    """Plot 2D scatter plot of clusters."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters, palette='tab10', s=100)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()

def main():
    file_path = '/Users/cheaulee/Desktop/mscproject/XGB/druggability_feature_table.tsv'  # Adjust the path as needed
    
    # Load and clean data
    data_cleaned, binary_columns, continuous_columns = load_and_clean_data(file_path)
    
    # Impute missing values
    data_cleaned = impute_missing_values(data_cleaned, continuous_columns)
    
    # Remove outliers
    data_no_outliers = remove_outliers_zscore(data_cleaned, continuous_columns)
    remaining_indices = data_no_outliers.index
    
    # Apply rank-based inverse transformation
    transformed_continuous = rank_transform(data_no_outliers, continuous_columns)
    
    # Combine binary and transformed continuous features
    features_transformed = transformed_continuous.join(data_no_outliers[binary_columns].reset_index(drop=True))
    features_transformed = features_transformed.dropna()
    final_indices = features_transformed.index
    
    # Determine optimal k
    k_range = range(2, 11)
    wcss, silhouette_scores, dbi_scores = determine_optimal_k(features_transformed, k_range)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(k_range, wcss, silhouette_scores, dbi_scores)
    
    # Set k to 3
    k = 3
    
    # Perform PCA
    features_pca = perform_pca(features_transformed)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features_pca)
    
    # Add the cluster assignments to the genes using the correct indices
    results = pd.DataFrame({'Gene_Name': data_cleaned.loc[final_indices, 'Gene_Name'].values, 'Cluster': clusters + 1})  # Adjust cluster numbering to start from 1
    
    # Plot clusters
    plot_clusters_2d(features_pca, results['Cluster'], 'K-Means Clusters on Druggability Features')
    
    # Print the first few rows of the results dataframe
    print(results.head())
    
    # Save the clustered dataframe
    output_clustered_file = '/Users/cheaulee/Desktop/mscproject/results/druggability_kmeans.csv'
    results.to_csv(output_clustered_file, index=False)
    
    print(f"Clustered genes with k={k} have been saved to {output_clustered_file}")

if __name__ == "__main__":
    main()

