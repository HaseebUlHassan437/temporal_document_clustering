"""
Clustering Module
Author: Haseeb Ul Hassan (MSCS25003)

This module handles:
- TF-IDF vectorization
- K-means clustering
- Dimensionality reduction for visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import json

class DocumentClusterer:
    """Clusters documents using TF-IDF and K-means"""
    
    def __init__(self, n_clusters=8, max_features=1000, random_state=42):
        """
        Initialize the clusterer
        
        Args:
            n_clusters (int): Number of clusters for K-means
            max_features (int): Maximum number of features for TF-IDF
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize models
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=0.8,  # Ignore terms that appear in >80% of documents
            min_df=5,    # Ignore terms that appear in <5 documents
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        self.pca = PCA(n_components=2, random_state=random_state)
        
        # Will store fitted models
        self.tfidf_matrix = None
        self.cluster_labels = None
        self.pca_coords = None
        self.top_terms_per_cluster = {}
    
    def fit(self, documents):
        """
        Fit the clustering model
        
        Args:
            documents (list or pd.Series): List of preprocessed text documents
            
        Returns:
            self
        """
        print("\n" + "="*60)
        print("STARTING CLUSTERING PROCESS")
        print("="*60)
        
        # Step 1: TF-IDF Vectorization
        print(f"\n[1/4] Creating TF-IDF vectors...")
        print(f"      - Max features: {self.max_features}")
        print(f"      - Number of documents: {len(documents):,}")
        
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        print(f"      ✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"      ✓ Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        # Step 2: K-means Clustering
        print(f"\n[2/4] Applying K-means clustering...")
        print(f"      - Number of clusters: {self.n_clusters}")
        
        self.cluster_labels = self.kmeans.fit_predict(self.tfidf_matrix)
        
        print(f"      ✓ Clustering complete")
        print(f"      ✓ Inertia: {self.kmeans.inertia_:.2f}")
        
        # Print cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"\n      Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"        Cluster {cluster_id}: {count:,} documents ({count/len(documents)*100:.1f}%)")
        
        # Step 3: Extract top terms per cluster
        print(f"\n[3/4] Extracting top terms for each cluster...")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(self.n_clusters):
            # Get the center of this cluster
            center = self.kmeans.cluster_centers_[cluster_id]
            
            # Get top 10 terms
            top_indices = center.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            self.top_terms_per_cluster[cluster_id] = top_terms
            
            print(f"      Cluster {cluster_id}: {', '.join(top_terms[:5])}...")
        
        # Step 4: PCA for visualization
        print(f"\n[4/4] Reducing dimensions for visualization (PCA)...")
        
        self.pca_coords = self.pca.fit_transform(self.tfidf_matrix.toarray())
        
        explained_var = self.pca.explained_variance_ratio_
        print(f"      ✓ PCA complete")
        print(f"      ✓ Explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%}")
        
        print("\n" + "="*60)
        print("✓ CLUSTERING COMPLETE")
        print("="*60)
        
        return self
    
    def get_cluster_results(self):
        """
        Get clustering results
        
        Returns:
            dict: Dictionary with cluster labels, PCA coordinates, and top terms
        """
        return {
            'cluster_labels': self.cluster_labels,
            'pca_x': self.pca_coords[:, 0],
            'pca_y': self.pca_coords[:, 1],
            'top_terms': self.top_terms_per_cluster
        }
    
    def save_model(self, output_dir):
        """
        Save the trained models
        
        Args:
            output_dir (str or Path): Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving models to {output_dir}...")
        
        # Save vectorizer
        with open(output_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save kmeans
        with open(output_dir / 'kmeans.pkl', 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        # Save PCA
        with open(output_dir / 'pca.pkl', 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Save top terms as JSON
        with open(output_dir / 'top_terms.json', 'w') as f:
            json.dump(self.top_terms_per_cluster, f, indent=2)
        
        print("✓ Models saved successfully!")
    
    def load_model(self, model_dir):
        """
        Load trained models
        
        Args:
            model_dir (str or Path): Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        print(f"Loading models from {model_dir}...")
        
        with open(model_dir / 'vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(model_dir / 'kmeans.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)
        
        with open(model_dir / 'pca.pkl', 'rb') as f:
            self.pca = pickle.load(f)
        
        with open(model_dir / 'top_terms.json', 'r') as f:
            self.top_terms_per_cluster = json.load(f)
        
        print("✓ Models loaded successfully!")


def process_and_cluster(input_csv, output_csv, n_clusters=8, max_features=1000):
    """
    Main function to process and cluster documents
    
    Args:
        input_csv (str): Path to preprocessed CSV file
        output_csv (str): Path to save clustered results
        n_clusters (int): Number of clusters
        max_features (int): Max TF-IDF features
    """
    # Load preprocessed data
    print(f"\nLoading preprocessed data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df):,} documents")
    
    # Initialize clusterer
    clusterer = DocumentClusterer(
        n_clusters=n_clusters,
        max_features=max_features
    )
    
    # Fit the model
    clusterer.fit(df['processed_text'])
    
    # Get results
    results = clusterer.get_cluster_results()
    
    # Add results to dataframe
    df['cluster'] = results['cluster_labels']
    df['pca_x'] = results['pca_x']
    df['pca_y'] = results['pca_y']
    
    # Save results
    print(f"\nSaving clustered data to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("✓ Saved successfully!")
    
    # Save models
    model_dir = Path(output_csv).parent / 'models'
    clusterer.save_model(model_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"Total documents: {len(df):,}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Output saved to: {output_csv}")
    print(f"Models saved to: {model_dir}")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).parent.parent
    
    INPUT_CSV = project_root / "data/processed_emails.csv"
    OUTPUT_CSV = project_root / "data/clustered_emails.csv"
    
    # Process and cluster
    df = process_and_cluster(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        n_clusters=8,
        max_features=1000
    )
    
    print("\n✓ Clustering module test complete!")