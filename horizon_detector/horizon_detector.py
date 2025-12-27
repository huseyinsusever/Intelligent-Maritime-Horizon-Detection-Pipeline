import cv2
import numpy as np
#load picture
img = cv2.imread('east_australian.jpg')

#Convert to HSV format. This is the best format for color analysis
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Specify the blue color range. Set for the sea
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create the mask, only the blues will remain white.
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Combine the mask with the original image.
blue_only = cv2.bitwise_and(img, img, mask=mask)

# Remove the edges of the masked blue area
edges = cv2.inRange(hsv, lower_blue, upper_blue)

# Find straight lines mathematically with HoughLinesP.
# This function returns the start and end coordinates of the lines.
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100, minLineLength=200, maxLineGap=10)

if lines is not None:

    x1, y1, x2, y2 = lines[0][0]

    cv2.putText(img, "Detected Horizon", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Blue Masking', blue_only)
cv2.imshow('Canny Analysis', edges)
cv2.imshow('Horizon Detection', img)

cv2.waitKey(0)
cv2.destroyAllWindows()