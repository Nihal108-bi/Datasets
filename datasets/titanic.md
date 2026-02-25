# Titanic Dataset

## Overview

The Titanic dataset contains information about passengers on the RMS Titanic, which sank on April 15, 1912. This dataset is widely used for binary classification tasks to predict passenger survival based on various features.

## Dataset Information

- **Samples**: ~891 (training set), ~418 (test set) on Kaggle
- **Features**: 11 features (mix of numerical and categorical)
- **Target**: Binary (Survived: 0 = No, 1 = Yes)
- **Task Type**: Binary classification
- **Missing Values**: Yes (Age, Cabin, Embarked)

## Features

1. **PassengerId** - Unique identifier for each passenger
2. **Survived** - Survival status (0 = No, 1 = Yes) [Target variable]
3. **Pclass** - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
4. **Name** - Passenger name
5. **Sex** - Gender (male/female)
6. **Age** - Age in years
7. **SibSp** - Number of siblings/spouses aboard
8. **Parch** - Number of parents/children aboard
9. **Ticket** - Ticket number
10. **Fare** - Passenger fare
11. **Cabin** - Cabin number
12. **Embarked** - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Common Use Cases

- Binary classification practice
- Feature engineering demonstrations
- Handling missing data
- Data visualization and exploratory data analysis
- Kaggle competition practice

## Loading the Dataset

### Using seaborn

```python
import seaborn as sns
import pandas as pd

# Load the dataset
titanic = sns.load_dataset('titanic')

print(f"Shape: {titanic.shape}")
print(f"\nFirst few rows:\n{titanic.head()}")
print(f"\nMissing values:\n{titanic.isnull().sum()}")
```

### From Kaggle (CSV)

```python
import pandas as pd

# After downloading from Kaggle
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())
print(f"\nSurvival rate: {train_df['Survived'].mean():.2%}")
```

### Using pandas from URL

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

print(titanic.head())
```

## Example: Data Preprocessing and Classification

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
titanic = sns.load_dataset('titanic')

# Select features
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# Create working dataframe
df = titanic[features + [target]].copy()

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Drop rows with remaining missing values
df.dropna(inplace=True)

# Split features and target
X = df.drop(target, axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:\n{feature_importance}")
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
titanic = sns.load_dataset('titanic')

# Survival rate by class
plt.figure(figsize=(10, 5))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Age distribution by survival
plt.figure(figsize=(10, 5))
sns.histplot(data=titanic, x='age', hue='survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = titanic.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(titanic[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()
```

## Key Insights

- **Gender**: Women had a significantly higher survival rate (~74%) than men (~19%)
- **Class**: First-class passengers had higher survival rates than third-class
- **Age**: Children had higher survival rates than adults
- **Family**: Having 1-3 family members aboard increased survival chances

## Data Source

- **Kaggle Competition**: https://www.kaggle.com/c/titanic
- **Seaborn**: Built-in dataset
- **GitHub Datasets**: Various repositories with Titanic data

## Citation

```
Titanic: Machine Learning from Disaster
Kaggle Competition
https://www.kaggle.com/c/titanic
```

## Additional Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## License

Public Domain - Available through multiple sources
