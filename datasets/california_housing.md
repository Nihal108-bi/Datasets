# California Housing Dataset

## Overview

The California Housing dataset contains information from the 1990 California census. It's a regression dataset used for predicting median house values in California districts based on various demographic and geographic features.

## Dataset Information

- **Samples**: 20,640
- **Features**: 8 numerical features
- **Target**: Median house value (in $100,000s)
- **Task Type**: Regression
- **Missing Values**: None

## Features

1. **MedInc** - Median income in block group
2. **HouseAge** - Median house age in block group
3. **AveRooms** - Average number of rooms per household
4. **AveBedrms** - Average number of bedrooms per household
5. **Population** - Block group population
6. **AveOccup** - Average number of household members
7. **Latitude** - Block group latitude
8. **Longitude** - Block group longitude

## Target Variable

- **MedHouseVal** - Median house value for California districts (in $100,000s)

## Common Use Cases

- Regression algorithm testing and benchmarking
- Spatial data analysis
- Feature engineering demonstrations
- Real estate price prediction
- Geographic visualization

## Loading the Dataset

### Using scikit-learn

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
california = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset description:\n{df.describe()}")
```

### Using pandas with as_frame parameter

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load as DataFrame directly
california = fetch_california_housing(as_frame=True)
df = california.frame

print(df.head())
print(f"\nTarget statistics:\n{df['MedHouseVal'].describe()}")
```

## Example: Linear Regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
california = fetch_california_housing()
X, y = california.data, california.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': california.feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nFeature Importance:\n{feature_importance}")
```

## Example: Advanced Regression with Random Forest

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
california = fetch_california_housing()
X, y = california.data, california.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std()):.4f})")

# Test predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': california.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:\n{feature_importance}")
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load data
california = fetch_california_housing(as_frame=True)
df = california.frame

# Geographic scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df['Longitude'], df['Latitude'],
    c=df['MedHouseVal'], cmap='viridis',
    alpha=0.4, s=10
)
plt.colorbar(scatter, label='Median House Value ($100k)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California Housing Prices by Location')
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for idx, col in enumerate(df.columns):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].set_xlabel('Value')
    axes[row, col_idx].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plots for key features vs target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'Latitude', 'Longitude']

for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    axes[row, col].scatter(df[feature], df['MedHouseVal'], alpha=0.3, s=5)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('MedHouseVal')
    axes[row, col].set_title(f'MedHouseVal vs {feature}')

plt.tight_layout()
plt.show()
```

## Key Insights

- **MedInc (Median Income)**: Strongest predictor of house value (positive correlation)
- **Location**: Coastal areas generally have higher house values
- **HouseAge**: Moderate correlation with house value
- **AveRooms**: Positive correlation with house value
- **Geographic patterns**: Clear spatial clustering of prices

## Data Source

- **Scikit-learn**: Built-in dataset
- **Original Source**: 1990 U.S. Census data

## Citation

```
Pace, R. Kelley, and Ronald Barry. 
"Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.
```

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Original Paper](https://www.sciencedirect.com/science/article/abs/pii/S016794739600140X)

## Advantages over Boston Housing

- Larger dataset (20,640 vs 506 samples)
- More recent data (1990 vs 1970s)
- No ethical concerns with features
- Geographic coordinates enable spatial analysis
- Better for modern machine learning workflows

## Tips for Better Performance

1. Scale/normalize features before training
2. Consider polynomial features for non-linear relationships
3. Use geographic features (latitude/longitude) creatively
4. Try ensemble methods (Random Forest, Gradient Boosting)
5. Consider spatial autocorrelation in modeling

## License

Public Domain - Available through scikit-learn
