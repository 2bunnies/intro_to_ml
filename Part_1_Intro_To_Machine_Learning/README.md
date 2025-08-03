# Part 1 - ML Zero to Hero
# Intro to Machine Learning 
https://www.youtube.com/watch?v=KNAWp2S3w94&list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO&index=1

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
part_1_intro_to_ml.ipynb
```

**3. Install all necessary libraries (tensorflow and numpy):**
```
!pip install tensorflow numpy
```

**4. Import all necessary libraries (tensorflow and numpy):**
```
# Github: https://github.com/2bunnies/intro_to_ml
# Import all necessary libraries
import tensorflow
import numpy
```

**5. Build a machine learning model:**
```
# Build an AI model to predict: y = 2x + 1
# Define the model itself:
# - Dense: single layer
# - units: dimension of the output
# - input_shape: single value of x to predict y value
model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(units=1, input_shape=[1])])

# Model will make a guess of what the relationship between x and y using optimizer and loss functions
# to get closer and closer to the correct formula
# optimizer: generate new guess based on how good or bad the previous guess is (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
# loss: function to calculate how good or bad the guess is (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
model.compile(optimizer='sgd', loss='mean_squared_error')
```

**6. Data: x and y array**
```
# x and y datas: if x = -1.0, y = -3.0, if x = 0.0, y = -1.0, and so on
xs = numpy.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = numpy.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```

**7. Make the model guess, calculate how accurate that is, and optimize it for 500 times**
```
# Fit the Xs to the Ys, and try 500 times
model.fit(xs, ys, epochs=500)
```

**8. Now we have a trained model**
```
# Predict the y when x = 10
# It will print a number very close to 19, 
# it shows that it is a high probability that is 19 but cannot be certain
print(model.predict(numpy.array([10.0])))
```