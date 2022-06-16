# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import requests
import shutil
import zipfile
import math
import tensorflow as tf
import torch
import skimage.exposure as exposure
import argparse


ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,help="Insert the path to input image")
args =  vars(ap.parse_args())
width, height = 800,800
image = cv2.imread(args["input"])
image = cv2.resize(image,(width,height)) # In order to resize the image 500x800


image2 = np.copy(image)



"""
Detection
- Grayscale and bluring
Canny Edge Detection
Hough transforms
Clustering intersections: Hierarchical clustering is used to group intersections by distance and average each group to create final coordinates (see below)
Final coordinates
"""

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Gaussian BLur

kernel_size = 3
gauss = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0,0)

# CAnny
low_threshold = 100
high_threshold = 150
edges = cv2.Canny(gauss, low_threshold, high_threshold)
cv2.imshow("Canny", edges)
cv2.waitKey(0)

#@title Hough Transform for Lines
# Define the Hough transform parameters
rho = 1
theta = np.pi/180 
threshold = 150
line_size = 1900 

# Make a copy the same size as our image to draw on
line_image = np.copy(image) #creating an image copy to draw lines on

# Run Hough on the edge-detected image
lines = cv2.HoughLines(edges, rho, theta, threshold)

# Iterate over the output "lines" and draw lines on the image copy, which
# requires to bring back the values from the hough space to the image space
if lines is not None:
  for line in lines:
    # Retrieve the rho and theta coordinates of the line found by houghLines
    rho,theta = line[0]

    # Convert polar coordinates to rectangular Coordinates
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    # Round off the values
    x1 = int(x0 + line_size*(-b))
    y1 = int(y0 + line_size*(a))
    x2 = int(x0 - line_size*(-b))
    y2 = int(y0 - line_size*(a))

    # cv2.line draws a line in line_image from the point(x1,y1) to (x2,y2). 
    # (0,0,255) denotes the colour of the line and 2 denotes the thickness.  
    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),2)


cv2.imshow("Lines",line_image)
cv2.waitKey(0)

"""Corner detection"""


# Detect corners (find the R values through an image)
dst = cv2.cornerHarris(gray, 5, 3, 0.04)

# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)

top_percentage =  0.01
ret, dst = cv2.threshold(dst,top_percentage*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
 
# Refine the corners using cv2.cornerSubPix()
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
 
#Convert the centroids and corners to integer
centroids = np.int0(centroids)
corners = np.int0(corners)

corner_image1 = np.copy(image2)
for j in range(0,centroids.shape[0]):
  cv2.circle( corner_image1, (centroids[j][0], centroids[j][1]), 3, (0,0,255), 1)

corner_image2 = np.copy(image2)
for j in range(0,corners.shape[0]):
  cv2.circle( corner_image2, (corners[j][0], corners[j][1]), 3, (0,0,255), 1)

cv2.imshow("centroids",corner_image1)
cv2.waitKey(0)


cv2.imshow("subp",corner_image2)
cv2.waitKey(0)


cv2.destroyAllWindows()

