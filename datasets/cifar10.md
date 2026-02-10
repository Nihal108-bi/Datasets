# CIFAR-10 Dataset

## Overview

CIFAR-10 (Canadian Institute For Advanced Research) is a widely-used dataset for image classification. It consists of 60,000 32x32 color images in 10 different classes, representing common objects.

## Dataset Information

- **Samples**: 60,000 (50,000 training, 10,000 testing)
- **Image Size**: 32x32 pixels (RGB color)
- **Features**: 3072 values (32x32x3)
- **Target**: 10 classes
- **Task Type**: Multi-class classification
- **Missing Values**: None

## Target Classes

1. **Airplane** (0)
2. **Automobile** (1)
3. **Bird** (2)
4. **Cat** (3)
5. **Deer** (4)
6. **Dog** (5)
7. **Frog** (6)
8. **Horse** (7)
9. **Ship** (8)
10. **Truck** (9)

## Common Use Cases

- Convolutional Neural Network (CNN) training
- Image classification benchmarking
- Data augmentation demonstrations
- Transfer learning practice
- Computer vision model evaluation

## Loading the Dataset

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Using PyTorch

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
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
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

## Example: CNN with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build CNN model
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Train model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Example: CNN with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define CNN architecture
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Prepare data with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

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

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display sample images
fig, axes = plt.subplots(5, 10, figsize=(15, 8))
for i in range(50):
    ax = axes[i // 10, i % 10]
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i][0]], fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(10, 5))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar([class_names[i] for i in unique], counts)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('CIFAR-10 Training Set - Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Data Augmentation Techniques

Common augmentation strategies for CIFAR-10:

- Random horizontal flips
- Random crops with padding
- Random rotation (±15 degrees)
- Color jittering
- Cutout/Random erasing
- Mixup

## Performance Benchmarks

- **Simple CNN**: ~70-75% accuracy
- **ResNet**: ~90-95% accuracy
- **State-of-the-art**: >99% accuracy (with advanced techniques)

## Data Source

- **Official Website**: https://www.cs.toronto.edu/~kriz/cifar.html
- **TensorFlow/Keras**: Built-in dataset
- **PyTorch**: Built-in dataset via torchvision

## Citation

```
Krizhevsky, A., & Hinton, G. (2009). 
Learning multiple layers of features from tiny images. 
Technical report, University of Toronto.
```

## Related Datasets

- **CIFAR-100**: Same format but with 100 classes
- **STL-10**: Larger images (96x96) with fewer samples
- **ImageNet**: Much larger dataset with 1000 classes

## Additional Resources

- [Official CIFAR Website](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Papers with Code - CIFAR-10](https://paperswithcode.com/dataset/cifar-10)
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Tips for Better Performance

1. Use data augmentation
2. Implement batch normalization
3. Use dropout for regularization
4. Try learning rate scheduling
5. Experiment with deeper architectures (ResNet, DenseNet)
6. Use pre-trained models for transfer learning

## License

Public Domain - Available for research and education
