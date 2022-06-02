# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import requests
import os
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



def crop_images(final_points,image3):
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

			Fcoord = np.array([final_points[k-1][j-1], final_points[k-1][j], final_points[k][j-1], final_points[k][j] ])
			ix = min(Fcoord[:,0])
			ex = max(Fcoord[:,0])
			iy = min(Fcoord[:,1])
			ey = max(Fcoord[:,1])
			kimg = image3[iy-60:ey+10,ix-10:ex+10]
			cv2.imshow("Mini Images",kimg)
			cv2.waitKey(0)
			filename = str("board"+str(k)+str(j)+".jpeg")
			print(filename)
			path = "Mini"
			cv2.imwrite(os.path.join(path,filename),kimg)
cv2.destroyAllWindows()
