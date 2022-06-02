# Libraries
import numpy as np

import cv2

import argparse


from warp_image import lines_image
from warp_image import intersection_points
from warp_image import cluster_points
from warp_image import corners
from segmentar_ord import crop_images


ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,help="Insert the path to input image")
args =  vars(ap.parse_args())
width, height = 800,800
image = cv2.imread(args["input"])
image = cv2.resize(image,(width,height)) # In order to resize the image 500x800

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
crop_images(final_points,image) # For keeping the cropped images it must be created a folder named "Mini" at your path

cv2.destroyAllWindows()
