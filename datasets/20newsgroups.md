# 20 Newsgroups Dataset

## Overview

The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. It's a popular dataset for text classification, clustering, and topic modeling experiments.

## Dataset Information

- **Samples**: ~20,000 documents
- **Features**: Text documents (variable length)
- **Target**: 20 newsgroup categories
- **Task Type**: Multi-class text classification
- **Missing Values**: None
- **Language**: English

## Target Classes (Newsgroups)

The 20 categories are organized into 6 broader themes:

### Computer Technology (5 classes)
- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`

### Recreation (4 classes)
- `rec.autos`
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`

### Science (4 classes)
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `sci.space`

### Politics (3 classes)
- `talk.politics.misc`
- `talk.politics.guns`
- `talk.politics.mideast`

### Religion (3 classes)
- `talk.religion.misc`
- `alt.atheism`
- `soc.religion.christian`

### For Sale (1 class)
- `misc.forsale`

## Common Use Cases

- Text classification
- Topic modeling
- Document clustering
- Feature extraction (TF-IDF, word embeddings)
- Multi-class classification practice
- Natural language processing demonstrations

## Loading the Dataset

### Using scikit-learn

```python
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Load all data
newsgroups = fetch_20newsgroups(subset='all')

print(f"Number of documents: {len(newsgroups.data)}")
print(f"Number of categories: {len(newsgroups.target_names)}")
print(f"\nCategories:\n{newsgroups.target_names}")

# Display first document
print(f"\nFirst document:\n{newsgroups.data[0][:500]}...")
print(f"Category: {newsgroups.target_names[newsgroups.target[0]]}")

# Load train/test split
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

print(f"\nTraining samples: {len(train.data)}")
print(f"Test samples: {len(test.data)}")
```

### Loading with Headers/Footers Removed

```python
from sklearn.datasets import fetch_20newsgroups

# Remove headers, footers, and quotes for cleaner text
newsgroups_clean = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

print(f"Cleaned document:\n{newsgroups_clean.data[0][:500]}...")
```

### Loading Specific Categories

```python
from sklearn.datasets import fetch_20newsgroups

# Load only specific categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroups_subset = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print(f"Number of documents: {len(newsgroups_subset.data)}")
print(f"Categories: {newsgroups_subset.target_names}")
```

## Example: Text Classification with TF-IDF and Naive Bayes

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Load data
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', MultinomialNB()),
])

# Train
print("Training model...")
text_clf.fit(train.data, train.target)

# Predict
print("Making predictions...")
predicted = text_clf.predict(test.data)

# Evaluate
accuracy = accuracy_score(test.target, predicted)
print(f"\nAccuracy: {accuracy:.4f}")

print(f"\nClassification Report:\n")
print(classification_report(test.target, predicted, target_names=test.target_names))

# Show confusion matrix for a subset of categories
print(f"\nTop-5 most confused category pairs:")
conf_matrix = confusion_matrix(test.target, predicted)
# Find top confused pairs (excluding diagonal)
np.fill_diagonal(conf_matrix, 0)
top_indices = np.argsort(conf_matrix.ravel())[-5:]
for idx in reversed(top_indices):
    i, j = np.unravel_index(idx, conf_matrix.shape)
    print(f"{test.target_names[i]} -> {test.target_names[j]}: {conf_matrix[i, j]} errors")
```

## Example: Classification with Logistic Regression

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load data
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000)),
])

# Parameter grid for tuning
parameters = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0],
}

# Grid search
print("Performing grid search...")
grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(train.data, train.target)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on test set
test_accuracy = grid_search.score(test.data, test.target)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## Example: Topic Modeling with LDA

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Load data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create document-term matrix
vectorizer = CountVectorizer(max_features=1000, stop_words='english', max_df=0.95, min_df=2)
doc_term_matrix = vectorizer.fit_transform(newsgroups.data)

# Fit LDA model
n_topics = 20
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(doc_term_matrix)

# Display top words for each topic
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10

print("Top words for each topic:")
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

## Example: Text Classification with Neural Networks (Keras)

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Tokenization
max_words = 10000
max_length = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train.data)

X_train = tokenizer.texts_to_sequences(train.data)
X_test = tokenizer.texts_to_sequences(test.data)

X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

y_train = train.target
y_test = test.target

# Build model
model = keras.Sequential([
    layers.Embedding(max_words, 128, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(20, activation='softmax')  # 20 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# Load data
newsgroups = fetch_20newsgroups(subset='all')

# Class distribution
plt.figure(figsize=(12, 6))
unique, counts = np.unique(newsgroups.target, return_counts=True)
plt.barh([newsgroups.target_names[i] for i in unique], counts)
plt.xlabel('Number of Documents')
plt.title('20 Newsgroups - Class Distribution')
plt.tight_layout()
plt.show()

# Document length distribution
doc_lengths = [len(doc.split()) for doc in newsgroups.data]

plt.figure(figsize=(10, 5))
plt.hist(doc_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Document Length (words)')
plt.ylabel('Frequency')
plt.title('Distribution of Document Lengths')
plt.axvline(np.mean(doc_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(doc_lengths):.0f}')
plt.axvline(np.median(doc_lengths), color='green', linestyle='--', label=f'Median: {np.median(doc_lengths):.0f}')
plt.legend()
plt.show()
```

## Performance Benchmarks

- **Naive Bayes with TF-IDF**: ~75-80% accuracy
- **Logistic Regression with TF-IDF**: ~80-85% accuracy
- **SVM with TF-IDF**: ~82-87% accuracy
- **LSTM/GRU**: ~80-85% accuracy
- **Pre-trained Transformers (BERT)**: ~90-93% accuracy

## Dataset Challenges

- **Class imbalance**: Some categories have more documents than others
- **Similar categories**: Some newsgroups are closely related (e.g., different computer or sport categories)
- **Noisy data**: Contains headers, footers, and quoted text from previous messages
- **Vocabulary size**: Large vocabulary can be computationally expensive

## Data Source

- **Original Source**: CMU (Carnegie Mellon University)
- **Scikit-learn**: Built-in dataset
- **Original collection**: Ken Lang

## Citation

```
Lang, K. (1995). Newsweeder: Learning to filter netnews. 
In Proceedings of the 12th international conference on machine learning (pp. 331-339).
```

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
- [Original Dataset Information](http://qwone.com/~jason/20Newsgroups/)
- [Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Tips for Better Performance

1. **Preprocessing**: Remove headers, footers, and quotes
2. **Feature Engineering**: Use TF-IDF instead of simple counts
3. **N-grams**: Consider bigrams and trigrams for better context
4. **Dimensionality Reduction**: Use techniques like LSA or LDA
5. **Ensemble Methods**: Combine multiple classifiers
6. **Deep Learning**: Use pre-trained embeddings or transformers
7. **Cross-validation**: Use proper validation to tune hyperparameters

## Common Preprocessing

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

## Related Tasks

- **Binary Classification**: Group categories into two classes (e.g., tech vs. non-tech)
- **Hierarchical Classification**: Use the natural hierarchy of newsgroups
- **Clustering**: Discover topics without using labels
- **Semi-supervised Learning**: Use unlabeled data to improve performance

## License

Public Domain - Available for research and educational purposes
