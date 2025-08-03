# Part 2 - ML Zero to Hero
# Basic Computer Vision with Machine Learning
https://www.youtube.com/playlist?list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO

## Tensorflow API Reference:
https://www.tensorflow.org/api_docs/python/tf

## Google Colabs
### Steps:

**1. Go to Google Colabs:**
https://colab.research.google.com/#create=true

**2. Rename the file to:**
```
part_2_flower.ipynb
```

**3. Make sure you have access to below directory, and save it to your "My Drive" directory:**
```
https://drive.google.com/drive/folders/1AkfoeMhkG_tN7qCr5sMdOMxKzgCykw1X?usp=drive_link
```

**4. Mount your Google Drive:**
```
from google.colab import drive
drive.mount('/content/drive')
```

**4. Install all necessary libraries:**
```
!pip install tensorflow matplotlib numpy Pillow
```

**5. Code: Explore dataset and create model:**
```
# SETUP
# -----------------------------------------------------------------------------------

# Import TensorFlow and other necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# EXPLORE THE DATASET
# -----------------------------------------------------------------------------------

# Dataset directory
data_dir = '/content/drive/MyDrive/ML/flower_photos'


# LOAD DATASET USING A KERAS UTILITY
# -----------------------------------------------------------------------------------

# Define some parameters for the loader

# batch_size: determines the number of samples processed in one forward and backward pass 
# during the training of a neural network. 
# It essentially dictates how many training examples are grouped together before the model's weights are updated.
batch_size = 32 

# Image size 180x180 pixels
img_height = 180
img_width = 180

# It's good practice to use a validation split when developing your model.
# Use 80% of the images for training and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# You can find the class names in the class_names attribute on these datasets. 
# These correspond to the directory names in alphabetical order.
class_names = train_ds.class_names
print(class_names)


# VISUALIZE THE DATA
# -----------------------------------------------------------------------------------
'''
# Here are the first nine images from the training dataset:
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# You will pass these datasets to the Keras Model.fit method for training later in this tutorial. 
# If you like, you can also manually iterate over the dataset and retrieve batches of images:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
'''

# CONFIGURE THE DATASET FOR PERFORMANCE
# -----------------------------------------------------------------------------------

# Make sure to use buffered prefetching, so you can yield data from disk without having I/O become blocking. 
# These are two important methods you should use when loading data:
# 1. Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. 
# This will ensure the dataset does not become a bottleneck while training your model. 
# If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# 2. Dataset.prefetch overlaps data preprocessing and model execution while training.
# Interested readers can learn more about both methods, as well as how to cache data to disk 
# in the Prefetching section of the Better performance with the tf.data API guide.
# https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# STANDARDIZE THE DATA
# -----------------------------------------------------------------------------------

# The RGB channel values are in the [0, 255] range. 
# This is not ideal for a neural network; in general you should seek to make your input values small.
# Here, you will standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling:
normalization_layer = layers.Rescaling(1./255)

# There are two ways to use this layer. You can apply it to the dataset by calling Dataset.map:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
# Or, you can include the layer inside your model definition, which can simplify deployment.


# CREATE THE MODEL
# -----------------------------------------------------------------------------------

# The Keras Sequential model consists of three convolution blocks (tf.keras.layers.Conv2D) 
# with a max pooling layer (tf.keras.layers.MaxPooling2D) in each of them. 
# There's a fully-connected layer (tf.keras.layers.Dense) with 128 units on top of it 
# that is activated by a ReLU activation function ('relu'). 
# This model has not been tuned for high accuracy; 
# the goal of this tutorial is to show a standard approach.
num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


# COMPILE THE MODEL
# -----------------------------------------------------------------------------------

# For this tutorial, choose the tf.keras.optimizers.Adam optimizer and 
# tf.keras.losses.SparseCategoricalCrossentropy loss function. 
# To view training and validation accuracy for each training epoch, 
# pass the metrics argument to Model.compile.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# MODEL SUMMARY
# -----------------------------------------------------------------------------------

# View all the layers of the network using the Keras Model.summary method:
model.summary()


# TRAIN THE MODEL
# -----------------------------------------------------------------------------------

# Train the model for 10 epochs with the Keras Model.fit method:
epochs=10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# VISUALIZE TRAINING RESULTS
# -----------------------------------------------------------------------------------

# Create plots of the loss and accuracy on the training and validation sets:

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# The plots show that training accuracy and validation accuracy are off by large margins, 
# and the model has achieved only around 60% accuracy on the validation set.
# The below tutorial sections show how to inspect what went wrong 
# and try to increase the overall performance of the model.
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

**6. Code: Address overfitting:**
```
# OVERFITTING
# -----------------------------------------------------------------------------------

# In the plots above, the training accuracy is increasing linearly over time, 
# whereas validation accuracy stalls around 60% in the training process. 
# Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of overfitting.

# When there are a small number of training examples, the model sometimes learns from noises or unwanted details 
# from training examples—to an extent that it negatively impacts the performance of the model on new examples. 
# This phenomenon is known as overfitting. It means that the model will have a difficult time generalizing on 
# a new dataset.

# There are multiple ways to fight overfitting in the training process. 
# In this tutorial, you'll use data augmentation and add dropout to your model.


# DATA AUGMENTATION
# -----------------------------------------------------------------------------------

# Overfitting generally occurs when there are a small number of training examples. 
# Data augmentation takes the approach of generating additional training data from your existing examples 
# by augmenting them using random transformations that yield believable-looking images. 
# This helps expose the model to more aspects of the data and generalize better.

# You will implement data augmentation using the following Keras preprocessing layers: 
# tf.keras.layers.RandomFlip, tf.keras.layers.RandomRotation, and tf.keras.layers.RandomZoom. 
# These can be included inside your model like other layers, and run on the GPU.
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Visualize a few augmented examples by applying data augmentation to the same image several times:
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# You will add data augmentation to your model before training in the next step.


# DROPOUT
# -----------------------------------------------------------------------------------

# Another technique to reduce overfitting is to introduce dropout regularization to the network.
# When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) 
# a number of output units from the layer during the training process. 
# Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. 
# This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.

# Create a new neural network with tf.keras.layers.Dropout before training it using the augmented images:
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])


# COMPILE AND TRAIN MODEL
# -----------------------------------------------------------------------------------

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# VISUALIZE TRAINING RESULTS
# -----------------------------------------------------------------------------------

# After applying data augmentation and tf.keras.layers.Dropout, there is less overfitting than before, 
# and training and validation accuracy are closer aligned:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

**7. Code: Predict on new data**
```
# PREDICT ON NEW DATA
# -----------------------------------------------------------------------------------

# Use your model to classify an image that wasn't included in the training or validation sets.
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```

**7. Code: Tensorflow Lite**
```
# USE TENSORFLOW LITE
# -----------------------------------------------------------------------------------

# TensorFlow Lite is a set of tools that enables on-device machine learning by helping developers 
# run their models on mobile, embedded, and edge devices.

# Convert the Keras Sequential model to a TensorFlow Lite model
# To use the trained model with on-device applications, first convert it to a smaller and 
# more efficient model format called a TensorFlow Lite model.

# In this example, take the trained Keras Sequential model and use tf.lite.TFLiteConverter.from_keras_model 
# to generate a TensorFlow Lite model:

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# The TensorFlow Lite model you saved in the previous step can contain several function signatures. 
# The Keras model converter API uses the default signature automatically. Learn more about TensorFlow Lite signatures.


# RUN THE TENSORFLOW LITE MODEL
# -----------------------------------------------------------------------------------

# You can access the TensorFlow Lite saved model signatures in Python via the tf.lite.Interpreter class.
# Load the model with the Interpreter:
TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

# Print the signatures from the converted model to obtain the names of the inputs (and outputs):
interpreter.get_signature_list()

# In this example, you have one default signature called serving_default. 
# In addition, the name of the 'inputs' is 'sequential_1_input', while the 'outputs' are called 'outputs'. 
# You can look up these first and last Keras layer names when running Model.summary, as demonstrated earlier in this tutorial.

# Now you can test the loaded TensorFlow Model by performing inference on a sample image with 
# tf.lite.Interpreter.get_signature_runner by passing the signature name as follows:
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite

# Similar to what you did earlier in the tutorial, you can use the TensorFlow Lite model to classify images 
# that weren't included in the training or validation sets.

# You have already tensorized that image and saved it as img_array. 
# Now, pass it to the first argument (the name of the 'inputs') of the loaded TensorFlow Lite model (predictions_lite), 
# compute softmax activations, and then print the prediction for the class with the highest computed probability.
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)

# The prediction generated by the lite model should be almost identical to the predictions generated by the original model:
print(np.max(np.abs(predictions - predictions_lite)))

# Of the five classes—'daisy', 'dandelion', 'roses', 'sunflowers', and 'tulips'—the model should predict the image 
# belongs to sunflowers, which is the same result as before the TensorFlow Lite conversion.
```