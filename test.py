import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# path  
path = r'./matthiostexture.png'
  
# Reading an image in default mode 
image = cv2.imread(path) 
# Window name in which image is displayed 
window_name = 'image'
  
# Using cv2.imshow() method  
# Displaying the image  
plt.imshow(image)
plt.show()
# cv2.imshow(window_name, image) 
# cv2.waitKey(0)