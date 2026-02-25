# Iris Dataset

## Overview

The Iris dataset is one of the most famous datasets in machine learning and statistics. It was introduced by British statistician and biologist Ronald Fisher in 1936. The dataset contains measurements of iris flowers from three different species.

## Dataset Information

- **Samples**: 150 (50 per class)
- **Features**: 4 numerical features
- **Target**: 3 classes (species)
- **Task Type**: Multi-class classification
- **Missing Values**: None

## Features

1. **Sepal Length** (cm) - Length of the sepal
2. **Sepal Width** (cm) - Width of the sepal
3. **Petal Length** (cm) - Length of the petal
4. **Petal Width** (cm) - Width of the petal

## Target Classes

- **Iris Setosa** (0)
- **Iris Versicolor** (1)
- **Iris Virginica** (2)

## Common Use Cases

- Introduction to machine learning classification
- Testing classification algorithms
- Visualization and exploratory data analysis
- Feature selection demonstrations
- Decision boundary visualization

## Loading the Dataset

### Using scikit-learn

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display basic information
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
```

### Using pandas (from CSV)

```python
import pandas as pd

# Load from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=column_names)

print(df.head())
```

### Using seaborn

```python
import seaborn as sns

# Load the dataset
iris = sns.load_dataset('iris')
print(iris.head())
```

## Example: Simple Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```

## Visualization Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
iris = sns.load_dataset('iris')

# Pairplot
sns.pairplot(iris, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Iris Dataset - Pairplot', y=1.02)
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
iris.boxplot(by='species', figsize=(12, 6))
plt.suptitle('Iris Dataset - Feature Distribution by Species')
plt.show()
```

## Data Source

- **Original Source**: Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/datasets/iris
- **Scikit-learn**: Built-in dataset
- **Seaborn**: Built-in dataset

## Citation

```
Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. 
Annual Eugenics, 7, Part II, 179-188.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
School of Information and Computer Science.
```

## Additional Resources

- [UCI ML Repository - Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Original Paper](https://digital.library.adelaide.edu.au/dspace/handle/2440/15227)

## License

Public Domain - Free to use for any purpose
