# Unsupervised Digit Clustering

An implementation of unsupervised clustering algorithms on the MNIST handritten digit dataset. The project compares hard and soft clustering approaches using K-means and Bernoulli Misture Models with Expectation-Maximization (EM) algorithm.

## Project Overview

This project explores unsupervised learning techniques for digit recognition without using ground truth labels. Two algorithms are implemented from scratch:

- **K-Means Clustering**: Hard clustering using Euclidean distance
- **Bernoulli Mixture Model**: Soft clustering using EM algorithm with binary data

The implementation achieves **55.9% purity** on MNIST using K-means, and **61.1% purity** using Bernoulli Mixture Models, demonstrating the ability to discover meaningful digit structure without supervision. Purity is reported for interpretability and is npt used as definitive performance metric for unsupervised learning.

## Features

- **Modular Architecture**: Clean separation of data loading, models, and visualization
- **Structured Notebooks**: Three demonstration notebooks with detailed analysis
- **Comprehensive Testing**: 39 unit tests with pytest covering edge cases
- **Pip-Installable Package**: Proper Python package structure with `setup.py`
- **Visualizations**: Cluster analysis, confusion matrices, soft assignment plots

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bvegaloj/unsupervised-digit-clustering.git
cd unsupervised-digit-clustering
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will automatically install all dependencies: `numpy`, `tensorflow`, `matplotlib`, `scikit-learn`, `pytest`.

## Usage

### Quick Start

```python
from src.data_loader import load_and_preprocess_mnist
from src.models.kmeans import KMeans
from src.models.bernoulli_mixture import BernoulliMixture
from src.visualization import plot_clusters

# Load data
X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
    n_samples=5000,
    binarize=False
)

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)
labels = kmeans.predict(X_test)

# Visualize results
plot_clusters(X_train, kmeans.labels_, title="K-Means Clustering Results")
```

### Bernoulli Mixture Model

```python
# Load binarized data for Bernoulli model
X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
    n_samples=5000,
    binarize=True
)

# Fit Bernoulli mixture
bmm = BernoulliMixture(n_components=10, random_state=42)
bmm.fit(X_train)

# Get soft assignments
probs = bmm.predict_proba(X_test)  # Shape: (n_samples, n_components)
hard_labels = bmm.predict(X_test)   # Argmax of probabilities
```

## Project Structure

```
unsupervised-digit-clustering/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # MNIST loading and preprocessing
│   ├── visualization.py         # Plotting functions
│   └── models/
│       ├── __init__.py
│       ├── kmeans.py            # K-means implementation
│       └── bernoulli_mixture.py # EM algorithm implementation
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py      # Data loading tests
│   ├── test_kmeans.py           # K-means tests
│   └── test_bernoulli_mixture.py # EM algorithm tests
├── notebooks/
│   ├── 01_data_exploration.ipynb      # MNIST exploration
│   ├── 02_kmeans_clustering.ipynb     # K-means analysis
│   └── 03_bernoulli_mixture.ipynb     # Soft clustering analysis
├── results/                     # Generated plots (gitignored)
├── setup.py                     # Package configuration
└── README.md
```

## Algorithms

### K-Means Clustering

**Hard clustering** algorithm that assigns each sample to exactly one cluster.

**Algorithm:**
1. Initialize K cluster centers randomly from data points
2. **E-step**: Assign each sample to nearest center (Euclidean distance)
3. **M-step**: Update centers as mean of assigned samples
4. Repeat until convergence (center shift < tolerance)

**Implementation details:**
- Squared Euclidean distance using expanded formula: `||x-c||² = ||x||² - 2x·c + ||c||²`
- Convergence check using Frobenius norm
- Handles empty clusters by retaining previous centers
- Supports edge case where `n_clusters > n_samples`

### Bernoulli Mixture Model

**Soft clustering** algorithm that assigns probability distributions over clusters.

**Algorithm:**
1. Initialize mixture weights π and component parameters θ randomly
2. **E-step**: Calculate posterior probabilities (responsibilities)
   - `p(z_k|x) ∝ π_k · p(x|θ_k)` where `p(x|θ_k) = ∏ θ_{kj}^{x_j} (1-θ_{kj})^{1-x_j}`
3. **M-step**: Update parameters using weighted statistics
   - `π_k = (1/N) Σ p(z_k|x_i)`
   - `θ_{kj} = Σ p(z_k|x_i)x_{ij} / Σ p(z_k|x_i)`
4. Repeat until convergence (log-likelihood change < tolerance)

**Implementation details:**
- Log-space computation for numerical stability
- Proper normalization in E-step (after all components calculated)
- Laplace smoothing (epsilon = 1e-10) to avoid log(0)

## Results

### K-Means Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **Purity** | 55.9% | 55.8% |
| **Iterations** | ~20 | - |
| **n_clusters** | 10 | 10 |

**Purity Calculation**: For each cluster, count the most frequent true digit label and sum across clusters.

The strong generalization (train ≈ test purity) indicates the algorithm discovered meaningful patterns in the data.

### Bernoulli Mixture Results

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **Purity** | 61.1% | - |
| **Iterations** | 496 | - |
| **Components Used** | Yes (all 10) | - |
| **Cluster Balance** | Well-distributed | - |

**Soft Clustering Advantages**:
- Captures uncertainty in ambiguous samples
- Entropy analysis reveals confident vs. uncertain predictions
- More nuanced than hard assignments
- **Higher purity than K-means** (61.1% vs 55.9%)

### Key Findings

1. **Both algorithms discover digit structure** without labels
2. **K-means favors separated clusters** (roundish digits like 0, 6)
3. **Soft clustering captures ambiguity** (e.g., 1 vs. 7, 3 vs. 8)
4. **Binarization helps Bernoulli model** but reduces grayscale information

## Demonstration Notebooks

### 01_data_exploration.ipynb
- MNIST dataset overview (60k train, 10k test)
- Digit variability analysis
- Binarization vs. grayscale comparison
- Sample visualizations

### 02_kmeans_clustering.ipynb
- K-means fitting and convergence analysis
- Cluster visualization with sample digits
- Confusion matrix (clusters vs. true labels)
- Purity calculation and interpretation

### 03_bernoulli_mixture.ipynb
- EM algorithm fitting process
- Soft vs. hard assignments comparison
- Entropy analysis for prediction confidence
- Uncertainty visualization (high-entropy samples)

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_kmeans.py -v
pytest tests/test_bernoulli_mixture.py -v
pytest tests/test_data_loader.py -v
```

**Test Coverage:**
- 39 total tests
- 10 data loader tests (preprocessing, types, edge cases)
- 14 K-means tests (clustering, convergence, edge cases)
- 17 Bernoulli mixture tests (EM algorithm, probabilities, edge cases)

**Edge cases tested:**
- Empty clusters (more clusters than samples)
- Single cluster/component
- Predict before fit (error handling)
- Single sample prediction
- Reproducibility with random seeds

## Dependencies

Core dependencies (automatically installed with `pip install -e .`):

- `numpy >= 1.19.0` - Numerical computations
- `tensorflow >= 2.0.0` - MNIST dataset loading
- `matplotlib >= 3.0.0` - Visualizations
- `scikit-learn >= 0.24.0` - Metrics and utilities
- `pytest >= 6.0.0` - Testing framework

## Technical Highlights

### Engineering practices
- Modular code structure with clear separation of concerns
- Algorithms implement from first principles to expose internal mechanics
- Type hints and comprehensive docstrings
- sklearn-style API (`fit()`, `predict()`) for familiarity and extensibility
- Reproducible results via explicit random seeds
- Numerical stability prioritized in EM via log-space computations
- Edge case handling (empty clusters, numerical stability)

### Mathematical Rigor
- Vectorized NumPy operations (no explicit loops where possible)
- Numerically stable implementations (log-space, Laplace smoothing)
- Proper probability normalization
- Convergence criteria based on standard metrics

### Documentation
- README with usage examples
- Narrative notebooks explaining methodology
- Comprehensive test suite
- Clean git history with descriptive commits

## Future Improvements

Potential enhancements for this project:

1. **Additional Algorithms**: Gaussian Mixture Models, DBSCAN, Hierarchical Clustering
2. **Hyperparameter Tuning**: Grid search for optimal K, tolerance, max_iters
3. **Evaluation Metrics**: Silhouette score, Davies-Bouldin index, adjusted Rand index
4. **Visualization**: t-SNE/UMAP embeddings, cluster dendrograms
5. **Performance**: Cython/Numba optimization for large-scale datasets

## Author

Daniel Vega
Artificial Intelligence and Machine Learning Enthusiast
- GitHub: [@bvegaloj](https://github.com/bvegaloj)

## License

MIT License

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- Clustering algorithms: Classic machine learning literature