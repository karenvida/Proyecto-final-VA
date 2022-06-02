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
#cv2.imshow("Canny", edges)
#cv2.waitKey(0)

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


#cv2.imshow("Lines",line_image)
#cv2.waitKey(0)

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

final_points = final_points[: len(final_points) - 9] 
corner_image3 = np.copy(image2)
plus = []
minus = []

for j in range(len(final_points)):
  cv2.circle( corner_image3, (final_points[j][0], final_points[j][1]), 3, (0,255,255), 1)
  plus.append(final_points[j][0]+final_points[j][1])
  minus.append(final_points[j][0]-final_points[j][1])

br,i1 = max(plus), plus.index(max(plus))
bl,i2 = max(minus), minus.index(max(minus))
tr,i3 = min(minus), minus.index(min(minus))
tl,i4 = min(plus), plus.index(min(plus))

# On a chessboard (8x8 squares) there are 9x9 intersection points, therefore, the intersection points are divided into 9 groups.

final_points.shape = (-1,9,2)

for k in range(len(final_points)):
	xcord = final_points[k,:,0]
	new_cord = np.zeros((9,2))
	#print(xcord)
	ran = len(xcord)
	for j in range(ran):
		mini, ind = min(xcord), np.where(xcord==(min(xcord)))[0][0]
		#print(mini)
		#print("ind: ", ind)
		xcord = np.delete(xcord,ind)
		#print(xcord)
		new_cord[j][0] = mini
		new_cord[j][1] = final_points[k][ind][1]
		
	final_points[k,:,:] = new_cord




# Take mini pictures
for k in range(1,len(final_points)):
	for j in range(1,9):
		image3 = image.copy()
		Fcoord = np.array([final_points[k-1][j-1], final_points[k-1][j], final_points[k][j-1], final_points[k][j] ])
		ix = min(Fcoord[:,0])
		ex = max(Fcoord[:,0])
		iy = min(Fcoord[:,1])
		ey = max(Fcoord[:,1])
		
		#cv2.circle(image3, final_points[k-1][j-1], 5, (255,255,0),3)
		#cv2.circle(image3, final_points[k-1][j], 5, (255,255,0),3)
		#cv2.circle(image3, final_points[k][j-1], 5, (255,255,0),3)
		#cv2.circle(image3, final_points[k][j], 5, (255,255,0),3)
		#cv2.circle(image3, (ix,iy), 5, (0,0,255),3)
		#cv2.circle(image3, (ex,ey), 5, (0,0,255),3)
		#cv2.imshow("Border, circles",image3)
		#cv2.waitKey(0)
		kimg = image3[iy-60:ey+10,ix-10:ex+10]
		cv2.imshow("Mini image",kimg)
		cv2.waitKey(0)


cv2.destroyAllWindows()

