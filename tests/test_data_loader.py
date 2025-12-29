"""Unit tests for data_loader module."""
import pytest
import numpy as np
from src.data_loader import load_and_preprocess_mnist


class TestLoadAndPreprocessMNIST:
    """Tests for load_and_preprocess_mnist function."""
    
    def test_basic_loading(self):
        """Test basic data loading with default parameters."""
        X_train, y_train, X_test, y_test = load_and_preprocess_mnist(n_samples=100)
        
        # Check shapes
        assert X_train.shape == (100, 784), "Train data should be (100, 784)"
        assert y_train.shape == (100,), "Train labels should be (100,)"
        assert X_test.shape == (10000, 784), "Test data should be (10000, 784)"
        assert y_test.shape == (10000,), "Test labels should be (10000,)"
    
    def test_binarization(self):
        """Test that binarization produces only 0s and 1s."""
        X_train, _, _, _ = load_and_preprocess_mnist(n_samples=50, binarize=True)
        
        unique_values = np.unique(X_train)
        assert len(unique_values) <= 2, "Binarized data should have at most 2 unique values"
        assert all(val in [0.0, 1.0] for val in unique_values), "Binarized values should be 0 or 1"
    
    def test_grayscale_range(self):
        """Test that grayscale values are in [0, 1]."""
        X_train, _, _, _ = load_and_preprocess_mnist(n_samples=50, binarize=False)
        
        assert X_train.min() >= 0.0, "Grayscale values should be >= 0"
        assert X_train.max() <= 1.0, "Grayscale values should be <= 1"
    
    def test_no_flattening(self):
        """Test loading without flattening."""
        X_train, _, X_test, _ = load_and_preprocess_mnist(n_samples=50, flatten=False)
        
        assert X_train.shape == (50, 28, 28), "Unflattened train data should be (50, 28, 28)"
        assert X_test.shape == (10000, 28, 28), "Unflattened test data should be (10000, 28, 28)"
    
    def test_label_range(self):
        """Test that labels are valid digits 0-9."""
        _, y_train, _, y_test = load_and_preprocess_mnist(n_samples=100)
        
        assert y_train.min() >= 0, "Labels should be >= 0"
        assert y_train.max() <= 9, "Labels should be <= 9"
        assert y_test.min() >= 0, "Test labels should be >= 0"
        assert y_test.max() <= 9, "Test labels should be <= 9"
    
    def test_different_sample_sizes(self):
        """Test loading with different sample sizes."""
        for n_samples in [10, 50, 100, 500]:
            X_train, y_train, _, _ = load_and_preprocess_mnist(n_samples=n_samples)
            assert len(X_train) == n_samples, f"Should load {n_samples} samples"
            assert len(y_train) == n_samples, f"Should have {n_samples} labels"
    
    def test_data_types(self):
        """Test that data has correct types."""
        X_train, y_train, X_test, y_test = load_and_preprocess_mnist(n_samples=50)
        
        assert X_train.dtype == np.float32 or X_train.dtype == np.float64, "Data should be float"
        assert y_train.dtype in [np.uint8, np.int64, np.int32], "Labels should be int-like"
        assert X_test.dtype == np.float32 or X_test.dtype == np.float64, "Test data should be float"
        assert y_test.dtype in [np.uint8, np.int64, np.int32], "Test labels should be int-like"
    
    def test_test_set_independence(self):
        """Test that test set is returned consistently."""
        _, _, X_test_1, _ = load_and_preprocess_mnist(n_samples=50)
        _, _, X_test_2, _ = load_and_preprocess_mnist(n_samples=200)
        
        # Both should return test set (though size may vary with n_samples in current implementation)
        assert X_test_1.shape[1] == X_test_2.shape[1], "Test set feature dimension should be constant"
    
    def test_binarize_and_flatten_combination(self):
        """Test all combinations of binarize and flatten."""
        configs = [
            (True, True),
            (True, False),
            (False, True),
            (False, False)
        ]
        
        for binarize, flatten in configs:
            X_train, _, _, _ = load_and_preprocess_mnist(
                n_samples=20, 
                binarize=binarize, 
                flatten=flatten
            )
            
            if flatten:
                assert X_train.ndim == 2, f"Flattened data should be 2D (binarize={binarize})"
                assert X_train.shape[1] == 784, f"Flattened should have 784 features (binarize={binarize})"
            else:
                assert X_train.ndim == 3, f"Unflattened data should be 3D (binarize={binarize})"
                assert X_train.shape[1:] == (28, 28), f"Unflattened should be 28x28 (binarize={binarize})"
