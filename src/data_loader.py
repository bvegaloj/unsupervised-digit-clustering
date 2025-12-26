# Libaries
import tensorflow
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist(n_samples=None, binarize=True, flatten=True):
    """Load MNIST dataset and preprocess it

    Args:
        n_samples (int, optional): Number of samples to load. Defaults uses 60,000.
        binarize (bool, optional): whether to binarize pixel values (threshold at 0.5). Defaults to True.
        flatten (bool, optional): whether to flatten images from 28x28 to 784D vectors. Defaults to True.
        
    Returns:
        X_train (np.array): preprocessed training data
        y_train (np.array): training labels
        X_test (np.array): preprocessed test data
        y_test (np.array): test labels
    """
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Subset training data if requested
    if n_samples is not None:
        X_train, y_train = X_train[:n_samples], y_train[:n_samples]
    
    # Apply same preprocessing to both train and test sets
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1) / 255  # Flatten and normalize
        X_test = X_test.reshape(X_test.shape[0], -1) / 255     # Flatten and normalize
    else:
        X_train = X_train / 255  # Just normalize
        X_test = X_test / 255    # Just normalize
    
    # Binarize if requested
    if binarize:
        X_train[X_train < 0.5] = 0
        X_train[X_train > 0.5] = 1
        X_test[X_test < 0.5] = 0
        X_test[X_test > 0.5] = 1
        
    return X_train, y_train, X_test, y_test