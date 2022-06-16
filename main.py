# Libraries
import numpy as np

import cv2

import argparse


from warp_image import lines_image
from warp_image import intersection_points
from warp_image import cluster_points
from warp_image import corners
from warp_image import reset
from segmentar_ord import crop_images
from comparision_f import per
from comparision_f import comp
from detection_pi import prediction


ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,help="Insert the path to input image")
#ap.add_argument("-i2", "--input2",required=True,help="Insert the path to input image")
args =  vars(ap.parse_args())
width, height = 800,800
image = cv2.imread(args["input"])
#imagec = cv2.imread(args["input2"])
image = cv2.resize(image,(width,height)) # In order to resize the image 500x800
#imagec = cv2.resize(imagec,(width,height)) # In order to resize the image 500x800


corner_image = np.copy(image)
corner_image2 = np.copy(image)
""" Identifying Hough Lines"""
lines = lines_image(image)
"""Intersections betweeen horizontal and vertical lines"""
intersected_points = intersection_points(lines)
"""CLustering close points"""
clusters = cluster_points(intersected_points)
final_points = np.int0(clusters)
"""Delet not necesary points"""
if(len(final_points)>81):
	final_points = final_points[: len(final_points) - 9]  # Elimina la ultima fila de puntos
"""Homography"""
final_warp = corners(final_points,image)

cv2.imshow("Perpendicular image",final_warp)
cv2.waitKey(0)

""" Cropping image"""
centers = crop_images(final_points,image) # For keeping the cropped images it must be created a folder named "Mini" at your path
for i in range(len(centers)):
	cv2.circle( corner_image, (centers[i][0], centers[i][1]), 3, (0,0,255), 1)
cv2.imshow("centroids",corner_image)
cv2.waitKey(0)
""" Detecting movement changes"""

board = reset()
print('New game!')
print(board)
pd=prediction(args["input"])
print('Detected pieces: ',len(pd))
#pieces.append([class_names[cls_id], cls_conf,x1,y1,x2,y2])
bases  = []
for i in range(len(pd)):
	#cv2.circle( corner_image2, (int(pd[i][2]), int(pd[i][3])), 3, (0,0,255), 1)
	#cv2.circle( corner_image2, (int(pd[i][4]), int(pd[i][5])), 3, (0,255,0), 1)
	x1 = pd[i][2]
	x2 = pd[i][4]
	y2 = pd[i][5]
	cv2.circle( corner_image2, (int((x1+x2)/2), int(y2)-20), 3, (0,255,0), 1)
	bases.append([int((x1+x2)/2),int(y2)-20])
cv2.imshow("centroids2",corner_image2)
cv2.waitKey(0)

cv2.destroyAllWindows()
