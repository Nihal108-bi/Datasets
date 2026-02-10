# IMDB Movie Reviews Dataset

## Overview

The IMDB dataset contains 50,000 movie reviews from the Internet Movie Database, split evenly into 25,000 reviews for training and 25,000 for testing. It's a binary sentiment classification dataset where the task is to determine whether a review is positive or negative.

## Dataset Information

- **Samples**: 50,000 (25,000 training, 25,000 testing)
- **Features**: Text reviews (variable length)
- **Target**: Binary sentiment (positive/negative)
- **Task Type**: Binary text classification (sentiment analysis)
- **Missing Values**: None
- **Language**: English

## Target Classes

- **0**: Negative sentiment (rating ≤ 4 out of 10)
- **1**: Positive sentiment (rating ≥ 7 out of 10)

## Common Use Cases

- Sentiment analysis demonstrations
- Natural language processing (NLP) practice
- Text classification benchmarking
- Learning word embeddings
- Testing deep learning models (LSTM, GRU, Transformers)
- Transfer learning with pre-trained language models

## Loading the Dataset

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the dataset (top 10,000 most frequent words)
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Average review length: {np.mean([len(x) for x in X_train]):.0f} words")

# Get word index
word_index = keras.datasets.imdb.get_word_index()

# Reverse word index to decode reviews
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode a review
def decode_review(encoded_review):
    # Note: indices are offset by 3 (0=padding, 1=start, 2=unknown)
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

print(f"\nFirst review (decoded):\n{decode_review(X_train[0])}")
print(f"Label: {'Positive' if y_train[0] == 1 else 'Negative'}")
```

### Using HuggingFace Datasets

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('imdb')

print(dataset)
print(f"\nTraining samples: {len(dataset['train'])}")
print(f"\nFirst review:\n{dataset['train'][0]['text']}")
print(f"Label: {'Positive' if dataset['train'][0]['label'] == 1 else 'Negative'}")
```

### Manual Download

```python
import pandas as pd

# You can also download from various sources
# Example: loading from CSV
url = "https://raw.githubusercontent.com/your-source/imdb.csv"
df = pd.read_csv(url)

print(df.head())
```

## Example: Simple LSTM Model with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and prepare data
vocab_size = 10000
max_length = 256

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad sequences to same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

# Build model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

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

## Example: Using Pre-trained Word Embeddings (GloVe)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load data
vocab_size = 10000
max_length = 256
embedding_dim = 100

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

# Load GloVe embeddings (you need to download GloVe first)
# Download from: https://nlp.stanford.edu/projects/glove/
embeddings_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
word_index = keras.datasets.imdb.get_word_index()
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build model with pre-trained embeddings
model = keras.Sequential([
    layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False  # Freeze embeddings
    ),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
```

## Example: Transformer Model with HuggingFace

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import tensorflow as tf

# Load dataset
dataset = load_dataset('imdb')

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare for training
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # Small subset for demo
test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))

# Convert to TensorFlow format
tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    batch_size=16,
    shuffle=True,
    tokenizer=tokenizer,
)

tf_test_dataset = model.prepare_tf_dataset(
    test_dataset,
    batch_size=16,
    shuffle=False,
    tokenizer=tokenizer,
)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=3)
```

## Example: Simple CNN for Text Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
vocab_size = 10000
max_length = 256

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

# Build CNN model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()

# Review length distribution
review_lengths = [len(x) for x in X_train]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(review_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Review Length (words)')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.axvline(np.mean(review_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(review_lengths):.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_train, bins=2, edgecolor='black', alpha=0.7)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Training Set - Class Distribution')
plt.xticks([0, 1], ['Negative', 'Positive'])

plt.tight_layout()
plt.show()

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.show()
```

## Performance Benchmarks

- **Bag of Words + Logistic Regression**: ~87-89% accuracy
- **Simple LSTM**: ~85-88% accuracy
- **Bidirectional LSTM**: ~88-90% accuracy
- **CNN**: ~88-90% accuracy
- **Pre-trained Transformers (BERT, RoBERTa)**: ~93-95% accuracy
- **State-of-the-art**: ~96%+ accuracy

## Dataset Statistics

- **Balanced Classes**: 50% positive, 50% negative reviews
- **Average Review Length**: ~230 words
- **Vocabulary Size**: ~88,000 unique words (full dataset)
- **Source**: Internet Movie Database (IMDb)

## Data Source

- **Stanford AI Lab**: http://ai.stanford.edu/~amaas/data/sentiment/
- **TensorFlow/Keras**: Built-in dataset
- **HuggingFace Datasets**: Available via `datasets` library
- **Original Paper**: Maas et al., 2011

## Citation

```
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). 
Learning word vectors for sentiment analysis. 
In Proceedings of the 49th annual meeting of the association for computational linguistics: 
Human language technologies (pp. 142-150).
```

## Additional Resources

- [Official Dataset Page](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Original Paper](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- [HuggingFace Dataset Card](https://huggingface.co/datasets/imdb)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)

## Tips for Better Performance

1. **Text Preprocessing**: Remove HTML tags, handle contractions
2. **Sequence Length**: Experiment with different max lengths (256, 512)
3. **Embeddings**: Use pre-trained embeddings (GloVe, FastText, Word2Vec)
4. **Architecture**: Try different models (LSTM, GRU, CNN, Transformers)
5. **Regularization**: Use dropout and early stopping
6. **Transfer Learning**: Fine-tune pre-trained language models (BERT, RoBERTa)
7. **Ensemble**: Combine multiple models for better results

## Common Preprocessing Steps

```python
import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

## License

The dataset is freely available for research purposes.
