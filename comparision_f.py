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
import sys
"""
ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,help="Insert the path to input image")
ap.add_argument("-i2", "--input2",required=True,help="Insert the path to input image")
args =  vars(ap.parse_args())
width, height = 800,800
image = cv2.imread(args["input"])
imagec = cv2.imread(args["input2"])
image = cv2.resize(image,(width,height)) # In order to resize the image 500x800
imagec = cv2.resize(imagec,(width,height)) # In order to resize the image 500x800
"""

def per(img):
	nb = np.sum(img == 0)
	nw = np.sum(img == 255)
	nnw = np.sum(img != 0)
	pdb = (nnw+nw)*100/(nw+nnw+nb)
	return pdb
def comp(image, imagec):
	comp = imagec - image
	frame_HSV = cv2.cvtColor(comp, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(frame_HSV, (10, 0, 200), (120, 150, 230))
	result = cv2.bitwise_and(comp, comp, mask=mask)
	cv2.imshow("wi",result)
	cv2.waitKey(0)



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

	pi = []
	pe = []
	por = 0
	# Take mini pictures
	for k in range(1,len(final_points)):
		for j in range(1,9):
			image3 = result.copy()
			Fcoord = np.array([final_points[k-1][j-1], final_points[k-1][j], final_points[k][j-1], final_points[k][j] ])
			ix = min(Fcoord[:,0])
			ex = max(Fcoord[:,0])
			iy = min(Fcoord[:,1])
			ey = max(Fcoord[:,1])
			kimg = image3[iy:ey,ix:ex]
			aux = per(kimg)
			if(aux > por):
				por = aux
				pi = pe.copy()
				pe.clear()
				pe.append(ix)
				pe.append(ex)
				pe.append(iy)
				pe.append(ey)
			
	print("Coordenadas iniciales: ",pi)
	print("Coordenadas finales: ",pe)	
cv2.destroyAllWindows()

