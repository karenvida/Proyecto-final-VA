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
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean

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

# Separate line into horizontal and vertical
def intersection_points(lines):
	h_lines, v_lines = [], []
	for line in lines:
		rho,theta = line[0]
		if (theta < np.pi / 4 or theta > np.pi - np.pi / 4):
			v_lines.append([rho, theta])
		else: 
			h_lines.append([rho, theta])
# Find the intersections of the lines
# def line_intersections(h_lines, v_lines):
	points = []
	for r_h, t_h in h_lines:
		for r_v, t_v in v_lines:
			a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
			b = np.array([r_h, r_v])
			inter_point = np.linalg.solve(a, b)
			points.append(inter_point)
	return np.array(points)
# SI las intersecciones de las lineas son iguales o similares a HArry corner, entonces son interseciones del tablero
intersected_points = intersection_points(lines)


def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])
clusters = cluster_points(intersected_points)
clusters = np.int0(clusters)

final_points = np.int0(clusters)
corner_image3 = np.copy(image2)
x_p = []
y_p = []
for j in range(len(final_points)):
  cv2.circle( corner_image3, (final_points[j][0], final_points[j][1]), 3, (0,255,255), 1)
  

cv2.imshow("Corners",corner_image3)
cv2.waitKey(0)

print(final_points)
"""Corner detection"""
"""

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

"""		


"""New method"""
corner_image4 = np.copy(image2)
cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4
cv2.drawContours(corner_image4, cnts, -1, (0,255,0), 2)
for c in cnts:
  epsilon = 0.01*cv2.arcLength(c,True)
  approx = cv2.approxPolyDP(c,epsilon,True)
  #print(len(approx))
  x,y,w,h = cv2.boundingRect(approx)
cv2.drawContours(corner_image4, [approx], 0, (0,255,0),2)
cv2.imshow('Corners 4',corner_image4)
cv2.waitKey(0)

cv2.destroyAllWindows()

