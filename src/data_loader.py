# Libaries
import tensorflow
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist(n_samples=None, binarize=True, flatten=True):
    """ Load MNIST dataset and preprocess it

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
    X_train, y_train = X_train[:n_samples], y_train[:n_samples]
    
    if flatten:
        X_train = X_train.reshape(n_samples, -1)    # Flatten images to 784D vectors
        X_train = X_train / 255                     # Normalize pixel values to [0, 1]
    
    # Binarize data for use in Bernoulli
    if binarize:
        X_train[X_train < 0.5] = 0
        X_train[X_train > 0.5] = 1
        
    return X_train, y_train, X_test, y_test