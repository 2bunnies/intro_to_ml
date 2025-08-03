# Part 2 - ML Zero to Hero
# Basic Computer Vision with Machine Learning
https://www.youtube.com/watch?v=bemDFpNooA8&list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO&index=2

## Github:
https://github.com/2bunnies/intro_to_ml

## Tensorflow API Reference:
https://www.tensorflow.org/api_docs/python/tf

## Google Colabs
### Steps:

**1. Go to Google Colabs:**
https://colab.research.google.com/#create=true

**2. Rename the file to:**
```
part_2_basic_computer_vision_with_ml.ipynb
```

**3. Install all necessary libraries (tensorflow, matplotlib, numpy):**
```
!pip install tensorflow matplotlib numpy
```

**4. Import all necessary libraries (tensorflow/keras):**
```
# Github: https://github.com/2bunnies/intro_to_ml
# Import all necessary libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy
```

**5. Load available fashion dataset from keras**
```
# Load available fashion MNIST dataset from keras
fashion_mnist = keras.datasets.fashion_mnist
```

# The label is the number indicating the class of that type of clothing:
```
# Class names and its corresponding number
# It uses number because:
# 1. Computer deals better with numbers
# 2. Avoid bias againsts other languages
class_names = [
    'T-shirt/top',  # 0
    'Trouser',      # 1
    'Pullover',     # 2
    'Dress',        # 3
    'Coat',         # 4
    'Sandal',       # 5
    'Shirt',        # 6
    'Sneaker',      # 7
    'Bag',          # 8
    'Ankle boot'    # 9
]

# Divide the dataset to 60,000 for training and 10,000 for test
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
```

**6. Build a machine learning model:**
```
# Build an AI model to predict item of clothing
# It acts like a filter which takes a 28x28 set of pixels and outputs 1 of 10 values
# activation function (https://www.tensorflow.org/api_docs/python/tf/keras/activations):
# tf.nn.relu: return a value if it is greater than 0, otherwise return a 0 (filtered out)
# tf.nn.softmax: picking the biggest number in a set
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 28x28 is the size of the images in the dataset
    keras.layers.Dense(128, activation=tf.nn.relu), # 128 functions to recognize the items of clothing
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 is the number of different items of clothing in our dataset
])
```

**7. Compile the model:**
```
# Model will make a guess of what the relationship between the train image datasets with the items of clothing using optimizer and loss functions to get closer and closer to the correct formula
# optimizer: generate new guess based on how good or bad the previous guess is (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
# loss: function to calculate how good or bad the guess is (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**8. Train the model:**
```
# Fit the train images to the train labels, and try 5 times
model.fit(train_images, train_labels, epochs=5)
```

**9. Test the model:**
```
# Use the 10,000 images that the model hasn't previously seen
# We use them to test how well our model performs by passing them to the evaluate model below
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

**10. Predict new image:**
```
# Predict test images
predictions = model.predict(test_images)

# Example: Predict the image in the test set
# Predict from test images
predictions = model.predict(test_images)

# Example: Predict the first image in the test set
which_test_image = 2
predicted_class = numpy.argmax(predictions[which_test_image])
print(f'\nPredicted class for the first test image: {class_names[predicted_class]}')
plt.imshow(test_images[which_test_image], cmap=plt.cm.binary)
plt.title(f'Actual: {class_names[test_labels[which_test_image]]}, Predicted: {class_names[predicted_class]}')
plt.show()
```