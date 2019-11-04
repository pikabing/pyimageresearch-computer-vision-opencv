import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 200, 255)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
	cv2.drawContours(output, [c], -1, (0, 0, 0), 2)

text = "I found {} objects".format(len(cnts))
cv2.putText(output, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
0.5, (34, 56, 78), 3)
cv2.imshow("Contours", output)
cv2.waitKey(0)




