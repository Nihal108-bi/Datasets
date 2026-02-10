# Boston Housing Dataset

## Overview

The Boston Housing dataset contains information about housing in the Boston area. It's a classic regression dataset used for predicting median home values based on various socioeconomic and geographic features.

## Dataset Information

- **Samples**: 506
- **Features**: 13 numerical features
- **Target**: Median value of owner-occupied homes (in $1000s)
- **Task Type**: Regression
- **Missing Values**: None
- **Note**: This dataset has been deprecated in scikit-learn 1.2+ due to ethical concerns about one of its features

## Features

1. **CRIM** - Per capita crime rate by town
2. **ZN** - Proportion of residential land zoned for lots over 25,000 sq.ft
3. **INDUS** - Proportion of non-retail business acres per town
4. **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX** - Nitric oxides concentration (parts per 10 million)
6. **RM** - Average number of rooms per dwelling
7. **AGE** - Proportion of owner-occupied units built prior to 1940
8. **DIS** - Weighted distances to five Boston employment centers
9. **RAD** - Index of accessibility to radial highways
10. **TAX** - Full-value property-tax rate per $10,000
11. **PTRATIO** - Pupil-teacher ratio by town
12. **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
13. **LSTAT** - % lower status of the population

## Target Variable

- **MEDV** - Median value of owner-occupied homes in $1000s

## Common Use Cases

- Regression algorithm testing
- Feature engineering demonstrations
- Predictive modeling practice
- Real estate price prediction
- Feature importance analysis

## Important Note

The Boston Housing dataset has been deprecated in scikit-learn (version 1.2+) due to ethical concerns regarding the "B" feature. Consider using alternative datasets like California Housing for regression tasks.

## Loading the Dataset

### Using scikit-learn (older versions < 1.2)

```python
from sklearn.datasets import load_boston
import pandas as pd

# Load dataset (only works in older sklearn versions)
boston = load_boston()

# Create DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nTarget statistics:\n{df['MEDV'].describe()}")
```

### Using pandas from CSV

```python
import pandas as pd

# Load from StatLib repository
url = "http://lib.stat.cmu.edu/datasets/boston"
# Or from a GitHub repository
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

df = pd.read_csv(url)
print(df.head())
```

### Alternative: California Housing Dataset

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# This is the recommended alternative
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(df.head())
```

## Example: Linear Regression

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data (using alternative method)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Separate features and target
X = df.drop('medv', axis=1)
y = df['medv']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nFeature Coefficients:\n{feature_importance}")
```

## Example: Advanced Regression with Gradient Boosting

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

X = df.drop('medv', axis=1)
y = df['medv']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(
    model, X_train_scaled, y_train, 
    cv=5, scoring='neg_mean_squared_error'
)

print(f"Cross-validation RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std()):.2f})")

# Test set predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.2f}")
print(f"Test R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:\n{feature_importance}")
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution of target variable
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['medv'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Median Home Value ($1000s)')
plt.ylabel('Frequency')
plt.title('Distribution of Home Values')

plt.subplot(1, 2, 2)
plt.boxplot(df['medv'])
plt.ylabel('Median Home Value ($1000s)')
plt.title('Box Plot of Home Values')
plt.tight_layout()
plt.show()

# Scatter plots of key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
important_features = ['rm', 'lstat', 'ptratio', 'indus', 'nox', 'dis']

for idx, feature in enumerate(important_features):
    row = idx // 3
    col = idx % 3
    axes[row, col].scatter(df[feature], df['medv'], alpha=0.5)
    axes[row, col].set_xlabel(feature.upper())
    axes[row, col].set_ylabel('MEDV')
    axes[row, col].set_title(f'MEDV vs {feature.upper()}')

plt.tight_layout()
plt.show()
```

## Key Insights

- **RM (rooms)**: Strong positive correlation with home value
- **LSTAT (lower status)**: Strong negative correlation with home value
- **PTRATIO (pupil-teacher ratio)**: Negative correlation with home value
- **CRIM (crime rate)**: Negative correlation with home value

## Data Source

- **Original Source**: Harrison, D. and Rubinfeld, D.L. (1978) "Hedonic prices and the demand for clean air"
- **StatLib**: http://lib.stat.cmu.edu/datasets/boston
- **UCI ML Repository**: Alternative sources available
- **GitHub**: Various repositories hosting the data

## Citation

```
Harrison, D. and Rubinfeld, D.L. (1978) 
Hedonic prices and the demand for clean air, 
J. Environ. Economics & Management, vol.5, 81-102.
```

## Ethical Considerations

The dataset contains a feature (B) that has been criticized for potential ethical issues. When using this dataset:
- Be aware of the historical context
- Consider using alternative datasets like California Housing
- If used for educational purposes, discuss the ethical implications
- Consider removing or not using the problematic feature

## Recommended Alternatives

1. **California Housing Dataset** - Similar regression task, more recent data
2. **Ames Housing Dataset** - More features, larger dataset
3. **House Prices: Advanced Regression Techniques** (Kaggle)

## Additional Resources

- [Scikit-learn Discussion on Deprecation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)
- [California Housing Alternative](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Original Paper](https://www.sciencedirect.com/science/article/abs/pii/0095069678900062)

## License

Public Domain - Available through multiple sources
