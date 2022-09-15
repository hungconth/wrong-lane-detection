# from detect_color_traffic_light import *
import numpy as np 
import cv2

# print(Predicted("0.jpg"))


image = cv2.imread(r"./img/IMG_0266.JPG", 1)
# Loading the image
image = cv2.resize(image, (1080, 720))
cv2.imshow("Frame", image)

# print(image.shape)
# im = image[300:400, :100]

# cv2.imshow("Frame1", im)
cv2.waitKey(0)
