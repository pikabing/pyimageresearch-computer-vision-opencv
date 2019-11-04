import numpy as np
import cv2
from transform import four_point_transform

image = cv2.imread("example_01.png")
pts = np.array(eval("[(73, 239), (356, 117), (475, 265), (187, 443)]"), dtype = "float32")

# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)

# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)