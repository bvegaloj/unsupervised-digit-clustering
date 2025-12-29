"""Unit tests for KMeans clustering algorithm."""
import pytest
import numpy as np
from src.models.kmeans import KMeans
from src.data_loader import load_and_preprocess_mnist


class TestKMeans:
    """Tests for KMeans clustering implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Load small dataset for testing."""
        X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
            n_samples=200, 
            binarize=False
        )
        return X_train, y_train, X_test, y_test
    
    def test_initialization(self):
        """Test KMeans object initialization."""
        model = KMeans(n_clusters=5, max_iters=100, random_state=42)
        
        assert model.n_clusters == 5
        assert model.max_iters == 100
        assert model.random_state == 42
        assert model.cluster_centers_ is None
        assert model.labels_ is None
    
    def test_fit_basic(self, sample_data):
        """Test basic fitting."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=10, random_state=42)
        model.fit(X_train)
        
        assert model.cluster_centers_ is not None
        assert model.labels_ is not None
        assert model.n_iter_ is not None
        assert model.cluster_centers_.shape == (10, 784)
        assert model.labels_.shape == (200,)
    
    def test_convergence(self, sample_data):
        """Test that algorithm converges within max_iters."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=5, max_iters=100, random_state=42)
        model.fit(X_train)
        
        assert model.n_iter_ <= 100, "Should converge within max_iters"
    
    def test_predict(self, sample_data):
        """Test prediction on new data."""
        X_train, _, X_test, _ = sample_data
        model = KMeans(n_clusters=10, random_state=42)
        model.fit(X_train)
        
        predictions = model.predict(X_test[:50])
        
        assert predictions.shape == (50,)
        assert predictions.min() >= 0
        assert predictions.max() < 10
    
    def test_predict_before_fit(self, sample_data):
        """Test that predict raises error before fitting."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=10)
        
        with pytest.raises((ValueError, AttributeError)):
            model.predict(X_train)
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict convenience method."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=10, random_state=42)
        
        labels = model.fit_predict(X_train)
        
        assert labels.shape == (200,)
        assert np.array_equal(labels, model.labels_)
    
    def test_reproducibility(self, sample_data):
        """Test that same random_state gives same results."""
        X_train, _, _, _ = sample_data
        
        model1 = KMeans(n_clusters=10, random_state=42)
        model1.fit(X_train)
        
        model2 = KMeans(n_clusters=10, random_state=42)
        model2.fit(X_train)
        
        assert np.allclose(model1.cluster_centers_, model2.cluster_centers_)
        assert np.array_equal(model1.labels_, model2.labels_)
    
    def test_different_n_clusters(self, sample_data):
        """Test with different numbers of clusters."""
        X_train, _, _, _ = sample_data
        
        for n_clusters in [2, 5, 10, 15]:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X_train)
            
            assert model.cluster_centers_.shape == (n_clusters, 784)
            assert model.labels_.max() == n_clusters - 1
            assert len(np.unique(model.labels_)) <= n_clusters
    
    def test_labels_in_range(self, sample_data):
        """Test that labels are valid cluster indices."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=5, random_state=42)
        model.fit(X_train)
        
        assert model.labels_.min() >= 0
        assert model.labels_.max() < 5
        assert model.labels_.dtype in [np.int32, np.int64]
    
    def test_cluster_centers_are_averages(self, sample_data):
        """Test that cluster centers are reasonable averages."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=5, random_state=42)
        model.fit(X_train)
        
        # Check that centers are within data range
        assert model.cluster_centers_.min() >= 0.0
        assert model.cluster_centers_.max() <= 1.0
    
    def test_empty_cluster_handling(self):
        """Test behavior with more clusters than samples."""
        # Only 5 samples but requesting 10 clusters - should handle gracefully
        X = np.random.rand(5, 10)
        
        model = KMeans(n_clusters=10, random_state=42, max_iters=20)
        labels = model.fit_predict(X)
        
        # Should still run without errors
        assert labels.shape == (5,)
        assert model.cluster_centers_.shape == (10, 10)
        # Not all clusters will be used
        assert len(np.unique(labels)) <= 5
    
    def test_single_cluster(self, sample_data):
        """Test with single cluster."""
        X_train, _, _, _ = sample_data
        model = KMeans(n_clusters=1, random_state=42)
        model.fit(X_train)
        
        assert np.all(model.labels_ == 0), "All samples should be in cluster 0"
        assert model.cluster_centers_.shape == (1, 784)
    
    def test_predict_single_sample(self, sample_data):
        """Test prediction on single sample."""
        X_train, _, X_test, _ = sample_data
        model = KMeans(n_clusters=10, random_state=42)
        model.fit(X_train)
        
        prediction = model.predict(X_test[0:1])
        
        assert prediction.shape == (1,)
        assert 0 <= prediction[0] < 10
