# Libraries
import numpy as np
from typing import Optional
from src.models.kmeans import KMeans

class BernoulliMixture:
    def __init__(
        self,
        n_components: int=10,
        max_iters: int=1000,
        tol: float=1e-6,
        random_state: Optional[int]=None
    ):
        """Mixture of Bernoulli distributions for binary data clustering
        
        Uses EM (Experctation-Maximization) algorithm to fit a mixture model 
        where each component is multivariate Bernoulli distribution
        
        Args:
            n_components (int, optional): Number of mixture components. Defaults to 10.
            max_iters (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Convergence tolerance. Defaults to 1e-6.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
        
        Attributes:
            mixture_weights_ (np.ndarray): Mixing coefficients π_k
            component_params_ (np.ndarray): Bernoulli parameters θ_k for each component
            labels_ (np.ndarray): Hard cluster assignments
            n_iter_: (int): Number of iterations run
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        # Set during fit()
        self.mixture_weights_ = None # π (n_components,)
        self.components_params_ = None # θ (n_components, n_features)
        self.labels_ = None
        self.n_iter_ = None
        
    def fit(self, X: np.ndarray) -> 'BernoulliMixture':
        """Fit Bernoulli mixture model using EM algorithm

        Args:
            X (np.ndarray): Training data

        Returns:
            BernoulliMixture (self) Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize mixture weights uniformly
        self.mixture_weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize component parameters randomly
        self.components_params_ = np.random.uniform(
            low=0.25,
            high=0.75,
            size=(self.n_components, n_features)
        )
        
        # EM iterations
        for i in range(self.max_iters):
            # E-step: calculate responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: update parameters
            old_params = self.components_params_.copy()
            self._m_step(X, responsibilities)
            
            # Check convergence
            param_change = np.linalg.norm(self.components_params_ - old_params)
            
            if param_change < self.tol:
                self.n_iter_ = i + 1
                break
            
        else:
            self.n_iter_ = self.max_iters

        # Store hard labels
        self.labels_ = self.predict(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard cluster assignment

        Args:
            X (np.ndarray): Data to predict

        Returns:
            labels (np.ndarray): Cluster label for each sample
        """
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities (soft assignment)

        Args:
            X (np.ndarray): Data to predict

        Returns:
            responsabilities (np.ndarray): Probability of each sample belonging to each component
        """
        if self.mixture_weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first")
        
        return self._e_step(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step

        Args:
            X (np.ndarray): Data to fit and predict

        Returns:
            BernoulliMixture (self): Fitted and predicted labels
        """
        return self.fit(X).predict(X)
    
    def _bernoulli_likelihood(self, X: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
        """Calculate Bernoulli likelihood for one component
        
        P(X | θ_k) = ∏_j θ_kj^x_j * (1-θ_kj)^(1-x_j)
        
        Args:
            X (np.ndarray): Data samples (n_samples, n_features)
            theta_k (np.ndarray): Parameters of the k-th component (n_features,)

        Returns:
            np.ndarray: Likelihood of each sample under the k-th component (n_samples,)
        """
        # Avoid log(0) by clipping theta
        theta_k = np.clip(theta_k, 1e-10, 1 - 1e-10)
        
        # Calculate log likelihood for numerical stability
        log_likelihood = np.sum(
            X * np.log(theta_k) + (1 - X) * np.log(1 - theta_k),
            axis=1
        )
        return np.exp(log_likelihood)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: Calculate responsibilities y_ik
        
        γ_ik = (π_k * P(x_i | θ_k)) / Σ_m (π_m * P(x_i | θ_m))

        Args:
            X (np.ndarray): Data samples (n_samples, n_features)

        Returns:
            responsabilities (np.ndarray): Responsibilities matrix (n_samples, n_components)
        """
        n_samples = X.shape[0]
        responsabilities = np.zeros((n_samples, self.n_components))
        
        # Calculate likelihood for each component
        for k in range(self.n_components):
            responsabilities[:, k] = (
                self.mixture_weights_[k] * 
                self._bernoulli_likelihood(X, self.components_params_[k])
            )
        
        # Normalize to get responsibilities (after all components calculated)
        responsabilities /= np.sum(responsabilities, axis=1, keepdims=True)
        
        return responsabilities
    
    def _m_step(self, X: np.ndarray, responsabilities: np.ndarray) -> None:
        """M-step: Update mixture weights π and component parameters θ.
    
        π_k = (Σ_i γ_ik) / N
        θ_k = (Σ_i γ_ik * x_i) / (Σ_i _ik)
        
        Args:
            X (np.ndarray): Data samples (n_samples, n_features)
            responsabilities (np.ndarray): Responsibilities matrix (n_samples, n_components)
        """
        n_samples = X.shape[0]
        
        # Update mixture weights
        self.mixture_weights_ = np.sum(responsabilities, axis=0) / n_samples
        
        # Update component parameters
        self.components_params_ = (
            (responsabilities.T @ X) / 
            np.sum(responsabilities, axis=0, keepdims=True).T
        )
        
        # Clip to avoid extreme values
        self.components_params_ = np.clip(self.components_params_, 1e-10, 1 - 1e-10)