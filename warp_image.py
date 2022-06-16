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
""" Segmentation by Hough Lines"""

def lines_image(image):

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

	return lines
		

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


def corners(final_points, image2):

	corner_image3 = np.copy(image2)
	plus = []
	minus = []

	for j in range(len(final_points)):
	  plus.append(final_points[j][0]+final_points[j][1])
	  minus.append(final_points[j][0]-final_points[j][1])

	br, i1 = max(plus), plus.index(max(plus))
	bl, i2 = max(minus), minus.index(max(minus))
	tr,i3 = min(minus), minus.index(min(minus))
	tl,i4 = min(plus), plus.index(min(plus))

	#cv2.imshow("Corners",corner_image3)
	#cv2.waitKey(0)
	trp = final_points[i2]
	tlp = final_points[i4]
	brp = final_points[i1]
	blp = final_points[i3]

	"""homography"""

	rect = np.array(((tlp[0]-4, tlp[1]-4), (trp[0]+4, trp[1]-4), (brp[0]-10, brp[1]-10),
	(blp[0]+5, blp[1]-5)), dtype="float32")
	width = 200
	height = 200
	dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-
	1]],dtype="float32")
	M = cv2.getPerspectiveTransform(rect,dst)
	warped_img = cv2.warpPerspective(image, M, (width, height))

	
	return warped_img
def reset():
	board = np.array([['R','P',0,0,0,0,'p','r'],
	['N','P',0,0,0,0,'p','n'],
	['B','P',0,0,0,0,'p','b'],
	['Q','P',0,0,0,0,'p','q'],
	['K','P',0,0,0,0,'p','k'],
	['B','P',0,0,0,0,'p','b'],
	['N','P',0,0,0,0,'p','n'],
	['R','P',0,0,0,0,'p','r'],])
	return board


