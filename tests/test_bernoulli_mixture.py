"""Unit tests for BernoulliMixture EM algorithm."""
import pytest
import numpy as np
from src.models.bernoulli_mixture import BernoulliMixture
from src.data_loader import load_and_preprocess_mnist


class TestBernoulliMixture:
    """Tests for Bernoulli Mixture Model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Load small binarized dataset for testing."""
        X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
            n_samples=200,
            binarize=True  # Bernoulli requires binary data
        )
        return X_train, y_train, X_test, y_test
    
    def test_initialization(self):
        """Test BernoulliMixture object initialization."""
        model = BernoulliMixture(n_components=5, max_iters=100, random_state=42)
        
        assert model.n_components == 5
        assert model.max_iters == 100
        assert model.random_state == 42
        assert model.mixture_weights_ is None
        assert model.components_params_ is None
        assert model.labels_ is None
    
    def test_fit_basic(self, sample_data):
        """Test basic fitting."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        model.fit(X_train)
        
        assert model.mixture_weights_ is not None
        assert model.components_params_ is not None
        assert model.labels_ is not None
        assert model.n_iter_ is not None
        assert model.mixture_weights_.shape == (10,)
        assert model.components_params_.shape == (10, 784)
        assert model.labels_.shape == (200,)
    
    def test_mixture_weights_sum_to_one(self, sample_data):
        """Test that mixture weights sum to 1."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=5, random_state=42)
        model.fit(X_train)
        
        assert np.isclose(model.mixture_weights_.sum(), 1.0), "Mixture weights should sum to 1"
        assert np.all(model.mixture_weights_ >= 0), "Mixture weights should be non-negative"
        assert np.all(model.mixture_weights_ <= 1), "Mixture weights should be <= 1"
    
    def test_component_params_in_range(self, sample_data):
        """Test that component parameters are valid probabilities."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=5, random_state=42)
        model.fit(X_train)
        
        assert np.all(model.components_params_ >= 0), "Parameters should be >= 0"
        assert np.all(model.components_params_ <= 1), "Parameters should be <= 1"
    
    def test_predict(self, sample_data):
        """Test hard assignment prediction."""
        X_train, _, X_test, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        model.fit(X_train)
        
        predictions = model.predict(X_test[:50])
        
        assert predictions.shape == (50,)
        assert predictions.min() >= 0
        assert predictions.max() < 10
    
    def test_predict_proba(self, sample_data):
        """Test soft assignment (probability) prediction."""
        X_train, _, X_test, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        model.fit(X_train)
        
        probs = model.predict_proba(X_test[:50])
        
        assert probs.shape == (50, 10)
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1 per sample"
        assert np.all(probs >= 0), "Probabilities should be non-negative"
        assert np.all(probs <= 1), "Probabilities should be <= 1"
    
    def test_predict_before_fit(self, sample_data):
        """Test that predict raises error before fitting."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=10)
        
        with pytest.raises(ValueError):
            model.predict(X_train)
    
    def test_predict_proba_before_fit(self, sample_data):
        """Test that predict_proba raises error before fitting."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=10)
        
        with pytest.raises(ValueError):
            model.predict_proba(X_train)
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict convenience method."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        
        labels = model.fit_predict(X_train)
        
        assert labels.shape == (200,)
        assert np.array_equal(labels, model.labels_)
    
    def test_reproducibility(self, sample_data):
        """Test that same random_state gives same results."""
        X_train, _, _, _ = sample_data
        
        model1 = BernoulliMixture(n_components=10, random_state=42, max_iters=50)
        model1.fit(X_train)
        
        model2 = BernoulliMixture(n_components=10, random_state=42, max_iters=50)
        model2.fit(X_train)
        
        assert np.allclose(model1.mixture_weights_, model2.mixture_weights_)
        assert np.allclose(model1.components_params_, model2.components_params_)
    
    def test_convergence(self, sample_data):
        """Test that algorithm converges within max_iters."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=5, max_iters=100, random_state=42)
        model.fit(X_train)
        
        assert model.n_iter_ <= 100, "Should converge within max_iters"
    
    def test_different_n_components(self, sample_data):
        """Test with different numbers of components."""
        X_train, _, _, _ = sample_data
        
        for n_components in [2, 5, 10]:
            model = BernoulliMixture(n_components=n_components, random_state=42, max_iters=50)
            model.fit(X_train)
            
            assert model.mixture_weights_.shape == (n_components,)
            assert model.components_params_.shape == (n_components, 784)
            assert len(np.unique(model.labels_)) <= n_components
    
    def test_predict_matches_argmax_proba(self, sample_data):
        """Test that hard assignments match argmax of soft assignments."""
        X_train, _, X_test, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        model.fit(X_train)
        
        hard_labels = model.predict(X_test[:50])
        soft_probs = model.predict_proba(X_test[:50])
        argmax_labels = np.argmax(soft_probs, axis=1)
        
        assert np.array_equal(hard_labels, argmax_labels), "Hard labels should match argmax of soft probs"
    
    def test_single_component(self, sample_data):
        """Test with single component."""
        X_train, _, _, _ = sample_data
        model = BernoulliMixture(n_components=1, random_state=42, max_iters=50)
        model.fit(X_train)
        
        assert np.all(model.labels_ == 0), "All samples should be in component 0"
        assert model.mixture_weights_[0] == 1.0, "Single component should have weight 1.0"
    
    def test_binary_data_requirement(self):
        """Test behavior with non-binary data (should still work but might not be ideal)."""
        # Create continuous data
        X = np.random.rand(100, 10)
        
        model = BernoulliMixture(n_components=3, random_state=42, max_iters=20)
        # Should not crash, though results may not be meaningful
        model.fit(X)
        
        assert model.mixture_weights_ is not None
        assert model.components_params_ is not None
    
    def test_tolerance_parameter(self):
        """Test that tolerance affects convergence."""
        X_train, _, _, _ = load_and_preprocess_mnist(n_samples=100, binarize=True)
        
        # Very loose tolerance should converge faster
        model_loose = BernoulliMixture(n_components=5, tol=1e-2, random_state=42)
        model_loose.fit(X_train)
        
        # Tight tolerance might take more iterations
        model_tight = BernoulliMixture(n_components=5, tol=1e-8, random_state=42)
        model_tight.fit(X_train)
        
        # Loose tolerance should converge in fewer or equal iterations
        assert model_loose.n_iter_ <= model_tight.n_iter_
    
    def test_predict_single_sample(self, sample_data):
        """Test prediction on single sample."""
        X_train, _, X_test, _ = sample_data
        model = BernoulliMixture(n_components=10, random_state=42)
        model.fit(X_train)
        
        prediction = model.predict(X_test[0:1])
        prob = model.predict_proba(X_test[0:1])
        
        assert prediction.shape == (1,)
        assert prob.shape == (1, 10)
        assert np.isclose(prob.sum(), 1.0)
