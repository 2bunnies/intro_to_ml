# Part 3 - ML Zero to Hero
# Introducing Convolutional Neural Networks
https://www.youtube.com/watch?v=bemDFpNooA8&list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO&index=3

## Github:
https://github.com/2bunnies/intro_to_ml

## Tensorflow API Reference:
https://www.tensorflow.org/api_docs/python/tf

## Google Colabs:
https://developers.google.com/codelabs/tensorflow-3-convolutions#0

### Steps:

**1. Go to Google Colabs:**
https://colab.research.google.com/#create=true

**2. Rename the file to:**
```
part_3_intro_convolutional_neural_network.ipynb
```

**3. Install all necessary libraries (tensorflow, matplotlib, numpy):**
```
!pip install tensorflow matplotlib numpy
```

**4. Code: Import necessary libraries**
```
# Start by importing some Python libraries:
import cv2
import numpy as np
from scipy import misc
```

**5. Code: Import the ascent picture**
```
i = misc.ascent()
```

**6. Code: See what the image looks like**
```
# Next, use the Pyplot library matplotlib to draw the image so that you know what it looks like:
import matplotlib.pyplot as plt
# You can see that it's an image of a stairwell. There are lots of features you can try and isolate. 
# For example, there are strong vertical lines.
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()
```

**7. Code: Create the convolution matrix/filter the image**
```
# The image is stored as a NumPy array, so we can create the transformed image by just copying that array. 
# The size_x and size_y variables will hold the dimensions of the image so you can loop over it later.
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# First, make a convolution matrix (or kernel) as a 3x3 array:

# This filter detects edges nicely
# It creates a filter that only passes through sharp edges and straight lines. 
# Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]] 
# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# Now, calculate the output pixels. Iterate over the image, leaving a 1-pixel margin, 
# and multiply each of the neighbors of the current pixel by the value defined in the filter.

# That means that the current pixel's neighbor above it and to the left of it will be multiplied by 
# the top-left item in the filter. Then, multiply the result by the weight and ensure that the result
# is in the range 0 through 255.

# Finally, load the new value into the transformed image:

for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        output_pixel = 0.0
        output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
        output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
        output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
        output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
        output_pixel = output_pixel + (i[x, y] * filter[1][1])
        output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
        output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
        output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
        output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
        output_pixel = output_pixel * weight
        if(output_pixel<0):
            output_pixel=0
        if(output_pixel>255):
            output_pixel=255
        i_transformed[x, y] = output_pixel
```

**8. Code: Examine the results**
```
# Now, plot the image to see the effect of passing the filter over it:
# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()   
```

**8. Code: Write code for pooling**
```
# The following code will show a (2, 2) pooling. Run it to see the output.
# You'll see that while the image is one-fourth the size of the original, it kept all the features.

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]
 
# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()
```