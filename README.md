# Common Datasets Repository

A curated collection of commonly used datasets for machine learning and data science projects. This repository serves as a centralized reference for dataset information, sources, and usage examples.

## 📚 Table of Contents

- [Overview](#overview)
- [Available Datasets](#available-datasets)
- [Dataset Categories](#dataset-categories)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains documentation and information about commonly used datasets in machine learning, data science, and statistical analysis. Each dataset includes:

- Detailed description
- Source and download links
- Features and attributes
- Common use cases
- Code examples for loading the data
- Related research papers or references

## 📊 Available Datasets

### Classification Datasets

1. **[Iris Dataset](datasets/iris.md)** - Classic dataset for classification, contains 150 samples of iris flowers with 4 features
2. **[Titanic Dataset](datasets/titanic.md)** - Binary classification dataset for survival prediction based on passenger information
3. **[MNIST](datasets/mnist.md)** - Handwritten digit recognition dataset with 70,000 grayscale images

### Regression Datasets

1. **[Boston Housing](datasets/boston_housing.md)** - Predict housing prices based on various features
2. **[California Housing](datasets/california_housing.md)** - Regression dataset for predicting house prices in California

### Computer Vision Datasets

1. **[CIFAR-10](datasets/cifar10.md)** - 60,000 32x32 color images in 10 classes
2. **[CIFAR-100](datasets/cifar100.md)** - Similar to CIFAR-10 but with 100 classes
3. **[Fashion-MNIST](datasets/fashion_mnist.md)** - Alternative to MNIST with fashion product images

### Natural Language Processing Datasets

1. **[IMDB Reviews](datasets/imdb.md)** - Sentiment analysis dataset with 50,000 movie reviews
2. **[20 Newsgroups](datasets/20newsgroups.md)** - Text classification dataset with ~20,000 documents

## 🗂️ Dataset Categories

- **Classification**: Datasets for supervised learning classification tasks
- **Regression**: Datasets for predicting continuous values
- **Computer Vision**: Image-based datasets for visual recognition tasks
- **NLP**: Text-based datasets for natural language processing
- **Time Series**: Sequential data for temporal analysis
- **Clustering**: Unlabeled datasets for unsupervised learning

## 💻 How to Use

Each dataset has its own detailed documentation file in the `datasets/` directory. Navigate to the specific dataset file to find:

1. Dataset description and background
2. Download/access instructions
3. Code examples in Python (using pandas, scikit-learn, TensorFlow, PyTorch)
4. Feature descriptions
5. Citation information

### Quick Start Example

```python
# Example: Loading Iris Dataset using scikit-learn
from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
```

## 🤝 Contributing

We welcome contributions! If you'd like to add a new dataset to this collection:

1. Fork this repository
2. Create a new dataset documentation file in the `datasets/` directory
3. Follow the template structure (see [CONTRIBUTING.md](CONTRIBUTING.md))
4. Update this README.md to include your dataset in the appropriate category
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🌟 Star This Repository

If you find this repository helpful, please consider giving it a star! It helps others discover these resources.

## 📧 Contact

For questions or suggestions, please open an issue in this repository.