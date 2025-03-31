import torch
import numpy as np
import os
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import tqdm

# Path to the folder containing your .pt files
embeddings_folder = 'outputs'

# Function to load all embeddings
def load_embeddings(folder_path):
    # Get all .pt files in the folder
    embedding_files = glob.glob(os.path.join(folder_path, '*.pt'))
    
    # Check if we found any files
    if len(embedding_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {folder_path}")
    
    print(f"Found {len(embedding_files)} embedding files")
    
    # Initialize list to store all embeddings
    all_embeddings = []
    file_names = []
    
    # Load each embedding file
    for file_path in tqdm(embedding_files, desc="Loading embeddings"):
        try:
            # Load the embedding tensor
            embedding = torch.load(file_path)
            
            # Check if the embedding has the expected shape [1, 512]
            if embedding.shape != (1, 512):
                print(f"Warning: File {file_path} has unexpected shape {embedding.shape}, skipping")
                continue
            
            # Convert to numpy and flatten if needed
            embedding_np = embedding.numpy().squeeze()
            all_embeddings.append(embedding_np)
            file_names.append(os.path.basename(file_path))
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Stack all the embeddings into a single numpy array
    if len(all_embeddings) == 0:
        raise ValueError("No valid embeddings were loaded")
    
    embeddings_array = np.vstack(all_embeddings)
    return embeddings_array, file_names

# Function to perform PCA
def perform_pca(embeddings):
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    
    # Perform dimensionality reduction
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by the two components: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.4f}")
    
    return reduced_embeddings, explained_variance

# Function to determine optimal number of clusters using silhouette score
def find_optimal_clusters(reduced_embeddings, max_clusters=10):
    silhouette_scores = []
    
    # Test different numbers of clusters from 2 to max_clusters
    range_n_clusters = range(2, min(max_clusters + 1, len(reduced_embeddings)))
    
    for n_clusters in tqdm(range_n_clusters, desc="Finding optimal clusters"):
        # Initialize KMeans with n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Fit KMeans
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(reduced_embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.4f}")
    
    # Find the optimal number of clusters
    optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, 'o-')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--')
    plt.title('Silhouette Score Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(alpha=0.3)
    plt.savefig('optimal_clusters_silhouette.png', dpi=300, bbox_inches='tight')
    
    return optimal_clusters

# Function to perform clustering and visualize
def cluster_and_visualize(reduced_embeddings, file_names, explained_variance, n_clusters=None):
    # If n_clusters is not provided, find optimal number
    if n_clusters is None:
        n_clusters = find_optimal_clusters(reduced_embeddings)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # Create a scatter plot with colors for different clusters
    plt.figure(figsize=(12, 10))
    
    # Create a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    # Plot each cluster with a different color
    for i in range(n_clusters):
        cluster_points = reduced_embeddings[cluster_labels == i]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            s=50, 
            c=[cmap(i)], 
            label=f'Cluster {i+1} ({np.sum(cluster_labels == i)} points)'
        )
    
    # Plot cluster centers
    plt.scatter(
        centers[:, 0], 
        centers[:, 1], 
        s=200, 
        c='black', 
        marker='X', 
        label='Cluster Centers'
    )
    
    plt.title(f'PCA Visualization with {n_clusters} Clusters')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.4f} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.4f} variance)')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the visualization
    plt.savefig('embedding_pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, centers

# Function to save the reduced embeddings with cluster information
def save_results(reduced_embeddings, file_names, cluster_labels, centers):
    # Create a dictionary to store the results
    results = {
        'file_names': file_names,
        'reduced_embeddings': reduced_embeddings,
        'cluster_labels': cluster_labels,
        'cluster_centers': centers
    }
    
    # Save the results as a numpy file for later use
    np.save('clustered_embeddings.npy', results)
    print("Clustered embeddings saved to 'clustered_embeddings.npy'")
    
    # Also save as a CSV for easy inspection
    result_df = np.column_stack((
        np.array(file_names, dtype=object),
        reduced_embeddings,
        cluster_labels[:, np.newaxis]
    ))
    
    # Save to CSV using numpy
    header = "filename,pc1,pc2,cluster"
    np.savetxt(
        'clustered_embeddings.csv', 
        result_df, 
        delimiter=',', 
        header=header, 
        fmt='%s'
    )
    print("Results also saved to 'clustered_embeddings.csv'")

# Main execution
def main():
    try:
        # Load all embeddings
        print("Loading embeddings from the outputs folder...")
        embeddings, file_names = load_embeddings(embeddings_folder)
        
        print(f"Successfully loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions each")
        
        # Perform PCA
        print("Performing PCA dimensionality reduction...")
        reduced_embeddings, explained_variance = perform_pca(embeddings)
        
        # Allow user to specify number of clusters or find optimal
        use_optimal = input("Find optimal number of clusters automatically? (y/n): ").lower().strip() == 'y'
        
        if use_optimal:
            n_clusters = None
        else:
            n_clusters = int(input("Enter the number of clusters to use: "))
        
        # Perform clustering and visualize
        print("Performing clustering and visualization...")
        cluster_labels, centers = cluster_and_visualize(
            reduced_embeddings, 
            file_names, 
            explained_variance, 
            n_clusters
        )
        
        # Save the results
        save_results(reduced_embeddings, file_names, cluster_labels, centers)
        
        print("PCA and clustering completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()