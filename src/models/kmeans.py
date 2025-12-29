# Libraries
import numpy as np
from typing import Optional

class KMeans:
    """K-Means clustering implementation"""
    
    def __init__(
        self, 
        n_clusters: int = 10,
        max_iters: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None
    ):
        """Initialize K-Means clustering
        
        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            max_iters (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Convergence tolerance. Defaults to 1e-6.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        # Set during fit()
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Fit K-Means clustering to data
        
        Args:
            X (np.ndarray): Training data
            
        Returns:
            self (KMeans): Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centers by random selection
        # Handle case where n_clusters > n_samples
        n_init_clusters = min(self.n_clusters, n_samples)
        random_indices = np.random.permutation(n_samples)
        self.cluster_centers_ = X[random_indices[:n_init_clusters]].copy()
        
        # If we need more clusters than samples, duplicate some randomly
        if n_init_clusters < self.n_clusters:
            extra_centers = X[np.random.choice(n_samples, self.n_clusters - n_init_clusters)]
            self.cluster_centers_ = np.vstack([self.cluster_centers_, extra_centers])
        
        # Iterative optmimization
        for iteration in range(self.max_iters):
            # E-step: assign clusters
            distances = self._calc_distances(X)
            assignments = self._assign_clusters(distances)
            
            # M-step: update centers
            old_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = self._update_centers(X, assignments)
            
            # Check convergence (using Frobenius norm)
            center_shift = np.linalg.norm(old_centers - self.cluster_centers_)
            
            if center_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        # Store final labels
        self.labels_ = np.argmin(self._calc_distances(X), axis=1)    
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data
        
        Args:
            X (np.ndarray): Data to predict
            
        Returns:
            Labels (np.ndarray): Predicted cluster labels"""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        return self.fit(X).predict(X)
    
    def _calc_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate squared Euclidean distances between X and cluster centers"""
        return ((-2 * X.dot(self.cluster_centers_.T) + 
                 np.sum(np.multiply(self.cluster_centers_, self.cluster_centers_), axis=1).T).T +
                np.sum(np.multiply(X, X), axis=1)).T
    
    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """Assign samples to nearest cluster (one-hot encoding)"""
        labels = np.argmin(distances, axis=1)
        return np.eye(self.n_clusters)[labels]
    
    def _update_centers(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Update cluster centers based on current assignments.
        
        Handles empty clusters by keeping their previous centers.
        """
        cluster_sums = X.T.dot(assignments)  # (n_features, n_clusters)
        cluster_counts = np.sum(assignments, axis=0)  # (n_clusters,)
        
        # Handle empty clusters by keeping their previous centers
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if cluster_counts[k] > 0:
                new_centers[k] = cluster_sums[:, k] / cluster_counts[k]
            else:
                # Keep the old center for empty clusters
                new_centers[k] = self.cluster_centers_[k]
        
        return new_centers