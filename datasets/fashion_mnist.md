# Fashion-MNIST Dataset

## Overview

Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. It serves as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

## Dataset Information

- **Samples**: 70,000 (60,000 training, 10,000 testing)
- **Image Size**: 28x28 pixels (grayscale)
- **Features**: 784 pixels (28x28 flattened)
- **Target**: 10 classes (fashion categories)
- **Task Type**: Multi-class classification
- **Missing Values**: None

## Target Classes

0. **T-shirt/top**
1. **Trouser**
2. **Pullover**
3. **Dress**
4. **Coat**
5. **Sandal**
6. **Shirt**
7. **Sneaker**
8. **Bag**
9. **Ankle boot**

## Common Use Cases

- Drop-in replacement for MNIST
- More challenging classification task than MNIST
- Testing neural network architectures
- Benchmarking deep learning algorithms
- Fashion item recognition
- Transfer learning experiments

## Loading the Dataset

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Using PyTorch

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
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

# Class names
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Example: CNN with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training history
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

## Example: Simple Neural Network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define model
class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nTest Accuracy: {100 * correct / total:.2f}%')
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display samples from each class
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(10):
    # Get indices for class i
    class_indices = np.where(y_train == i)[0][:10]
    for j in range(10):
        ax = axes[i, j]
        ax.imshow(X_train[class_indices[j]], cmap='gray')
        if j == 0:
            ax.set_ylabel(class_names[i], fontsize=12)
        ax.axis('off')
plt.suptitle('Fashion-MNIST: 10 samples from each class', fontsize=16)
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(10, 5))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar([class_names[i] for i in unique], counts)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Fashion-MNIST Training Set - Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## Dataset Comparison: Fashion-MNIST vs MNIST

| Aspect | Fashion-MNIST | MNIST |
|--------|--------------|-------|
| Task Difficulty | More challenging | Easier |
| Image Content | Fashion items | Handwritten digits |
| Typical Accuracy | 88-95% | 95-99% |
| Real-world relevance | Higher | Lower |
| Class distinction | More subtle | More distinct |

## Performance Benchmarks

- **Simple Neural Network**: ~87-89% accuracy
- **CNN**: ~90-93% accuracy
- **Advanced CNN (ResNet, etc.)**: ~94-95% accuracy
- **State-of-the-art**: ~96%+ accuracy

## Data Source

- **Official Repository**: https://github.com/zalandoresearch/fashion-mnist
- **TensorFlow/Keras**: Built-in dataset
- **PyTorch**: Built-in dataset via torchvision

## Citation

```
Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
```

## Additional Resources

- [Official GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)
- [Research Paper](https://arxiv.org/abs/1708.07747)
- [Benchmark Results](https://github.com/zalandoresearch/fashion-mnist#benchmark)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/keras/classification)

## Why Use Fashion-MNIST?

1. **Drop-in replacement**: Same format as MNIST (28x28 grayscale)
2. **More challenging**: Better test of algorithm performance
3. **More relevant**: Fashion classification has real-world applications
4. **Research benchmark**: Widely used in academic papers
5. **Balanced classes**: Equal samples per class

## Tips for Better Performance

1. Use data augmentation (rotation, shift, zoom)
2. Implement batch normalization
3. Try deeper architectures
4. Use dropout for regularization
5. Experiment with learning rate scheduling
6. Consider ensemble methods

## License

MIT License - Available for research and commercial use
