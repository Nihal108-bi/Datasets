# CIFAR-100 Dataset

## Overview

CIFAR-100 is similar to CIFAR-10 but with 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes are grouped into 20 superclasses for a hierarchical classification structure.

## Dataset Information

- **Samples**: 60,000 (50,000 training, 10,000 testing)
- **Image Size**: 32x32 pixels (RGB color)
- **Features**: 3072 values (32x32x3)
- **Target**: 100 fine classes, 20 coarse classes
- **Task Type**: Multi-class classification
- **Missing Values**: None

## Class Structure

The dataset has a two-level hierarchy:
- **20 Superclasses (coarse labels)**: Broad categories
- **100 Classes (fine labels)**: Specific categories

### Superclasses and Their Classes

Each superclass contains 5 classes. Examples:

- **aquatic mammals**: beaver, dolphin, otter, seal, whale
- **fish**: aquarium fish, flatfish, ray, shark, trout
- **flowers**: orchids, poppies, roses, sunflowers, tulips
- **food containers**: bottles, bowls, cans, cups, plates
- **fruit and vegetables**: apples, mushrooms, oranges, pears, sweet peppers
- **household electrical devices**: clock, computer keyboard, lamp, telephone, television
- **household furniture**: bed, chair, couch, table, wardrobe
- **insects**: bee, beetle, butterfly, caterpillar, cockroach
- **large carnivores**: bear, leopard, lion, tiger, wolf
- **large man-made outdoor things**: bridge, castle, house, road, skyscraper
- **large natural outdoor scenes**: cloud, forest, mountain, plain, sea
- **large omnivores and herbivores**: camel, cattle, chimpanzee, elephant, kangaroo
- **medium-sized mammals**: fox, porcupine, possum, raccoon, skunk
- **non-insect invertebrates**: crab, lobster, snail, spider, worm
- **people**: baby, boy, girl, man, woman
- **reptiles**: crocodile, dinosaur, lizard, snake, turtle
- **small mammals**: hamster, mouse, rabbit, shrew, squirrel
- **trees**: maple, oak, palm, pine, willow
- **vehicles 1**: bicycle, bus, motorcycle, pickup truck, train
- **vehicles 2**: lawn-mower, rocket, streetcar, tank, tractor

## Common Use Cases

- Fine-grained image classification
- Hierarchical classification
- Transfer learning practice
- More challenging alternative to CIFAR-10
- Testing advanced CNN architectures

## Loading the Dataset

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
# For coarse labels: label_mode='coarse'

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f'Class: {y_train[i][0]}')
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

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR100(
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

## Example: CNN with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build model with more capacity for 100 classes
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(100, activation='softmax')
])

# Compile
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

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")
```

## Example: Hierarchical Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data with both fine and coarse labels
(X_train_fine, y_train_fine), (X_test_fine, y_test_fine) = keras.datasets.cifar100.load_data(label_mode='fine')
(X_train_coarse, y_train_coarse), (X_test_coarse, y_test_coarse) = keras.datasets.cifar100.load_data(label_mode='coarse')

X_train = X_train_fine.astype('float32') / 255.0
X_test = X_test_fine.astype('float32') / 255.0

# Build model with two outputs
input_layer = layers.Input(shape=(32, 32, 3))

# Shared layers
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Coarse classification head
coarse_output = layers.Dense(20, activation='softmax', name='coarse')(x)

# Fine classification head
fine_output = layers.Dense(100, activation='softmax', name='fine')(x)

# Create model
model = keras.Model(inputs=input_layer, outputs=[coarse_output, fine_output])

# Compile with multiple losses
model.compile(
    optimizer='adam',
    loss={
        'coarse': 'sparse_categorical_crossentropy',
        'fine': 'sparse_categorical_crossentropy'
    },
    loss_weights={'coarse': 0.3, 'fine': 0.7},
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    {'coarse': y_train_coarse, 'fine': y_train_fine},
    validation_data=(X_test, {'coarse': y_test_coarse, 'fine': y_test_fine}),
    epochs=50,
    batch_size=64
)
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

# Display random samples
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    ax = axes[i // 10, i % 10]
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx])
    ax.set_title(f'{y_train[idx][0]}', fontsize=8)
    ax.axis('off')
plt.suptitle('CIFAR-100: Random Samples', fontsize=16)
plt.tight_layout()
plt.show()
```

## Dataset Comparison: CIFAR-100 vs CIFAR-10

| Aspect | CIFAR-100 | CIFAR-10 |
|--------|-----------|----------|
| Classes | 100 | 10 |
| Samples per class | 600 | 6,000 |
| Task difficulty | More challenging | Easier |
| Typical accuracy | 60-80% | 90-95% |
| Hierarchical structure | Yes (20 superclasses) | No |

## Performance Benchmarks

- **Simple CNN**: ~40-50% accuracy
- **ResNet**: ~70-75% accuracy
- **Advanced architectures**: ~80-85% accuracy
- **State-of-the-art**: ~90%+ accuracy

## Challenges

- **Class imbalance in superclasses**: Each superclass has only 5 fine classes
- **Limited samples per class**: Only 500 training images per class
- **High inter-class similarity**: Many classes look similar (e.g., different types of flowers)
- **Small image size**: 32x32 makes fine distinctions difficult

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

## Additional Resources

- [Official CIFAR Website](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Papers with Code - CIFAR-100](https://paperswithcode.com/dataset/cifar-100)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100)

## Tips for Better Performance

1. Use strong data augmentation (rotation, flip, color jitter, cutout)
2. Use deeper architectures (ResNet, DenseNet, EfficientNet)
3. Implement learning rate scheduling
4. Consider transfer learning from ImageNet
5. Use techniques like mixup and cutmix
6. Leverage the hierarchical structure for better generalization
7. Use batch normalization and dropout
8. Train for many epochs (100+) with patience

## License

Public Domain - Available for research and education
