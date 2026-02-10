# MNIST Dataset

## Overview

The MNIST (Modified National Institute of Standards and Technology) database is a large dataset of handwritten digits commonly used for training and testing in the field of machine learning. It's a benchmark dataset for image classification tasks.

## Dataset Information

- **Samples**: 70,000 (60,000 training, 10,000 testing)
- **Image Size**: 28x28 pixels (grayscale)
- **Features**: 784 pixels (28x28 flattened)
- **Target**: 10 classes (digits 0-9)
- **Task Type**: Multi-class classification
- **Missing Values**: None

## Features

- **Input**: Grayscale images of handwritten digits
- **Pixel Values**: 0-255 (0 = white/background, 255 = black/foreground)
- **Format**: Typically normalized to 0-1 range for neural networks

## Target Classes

- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

## Common Use Cases

- Introduction to computer vision and deep learning
- Testing neural network architectures
- Benchmarking classification algorithms
- Learning convolutional neural networks (CNNs)
- Transfer learning demonstrations

## Loading the Dataset

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Label distribution: {np.bincount(y_train)}")

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Display a sample image
plt.figure(figsize=(6, 6))
plt.imshow(X_train[0], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.axis('off')
plt.show()
```

### Using PyTorch

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### Using scikit-learn

```python
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Convert to numpy arrays and normalize
X = np.array(X) / 255.0
y = np.array(y, dtype=int)
```

## Example: Simple Neural Network with Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Example: Simple Neural Network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'\nTest Accuracy: {100 * correct / total:.2f}%')
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display multiple samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Display class distribution
plt.figure(figsize=(10, 5))
plt.hist(y_train, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Digit Class')
plt.ylabel('Frequency')
plt.title('MNIST Training Set - Class Distribution')
plt.xticks(range(10))
plt.grid(axis='y', alpha=0.3)
plt.show()
```

## Dataset Statistics

- **Balanced Classes**: Each digit appears approximately 6,000-7,000 times in the training set
- **Image Quality**: Clean, centered digits on uniform background
- **Preprocessing**: Already size-normalized and centered

## Data Source

- **Official Website**: http://yann.lecun.com/exdb/mnist/
- **TensorFlow/Keras**: Built-in dataset
- **PyTorch**: Built-in dataset via torchvision
- **Scikit-learn**: Available via OpenML

## Citation

```
LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST handwritten digit database. 
AT&T Labs [Online]. Available: http://yann.lecun.com/exdb/mnist
```

## Additional Resources

- [Official MNIST Website](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow MNIST Tutorial](https://www.tensorflow.org/datasets/catalog/mnist)
- [PyTorch MNIST Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Research Papers Using MNIST](http://yann.lecun.com/exdb/publis/index.html#lecun-98)

## Related Datasets

- **Fashion-MNIST**: A drop-in replacement with fashion items instead of digits
- **EMNIST**: Extended MNIST with letters and digits
- **KMNIST**: Japanese character recognition dataset

## License

Public Domain - Created by Yann LeCun, Corinna Cortes, and Christopher Burges
