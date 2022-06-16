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


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel_size = 3
gauss = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0,0)

ret,th1 = cv2.threshold(gauss,110,255,cv2.THRESH_BINARY)

cv2.imshow("Thresholded", th1)
cv2.waitKey(0)


"""New method"""
img = np.copy(image)
cnts,_ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4

#cv2.drawContours(img, cnts, -1, (0,0,255), 2)

print('# Con: ', len(cnts))
for c in cnts:
  epsilon = 0.01*cv2.arcLength(c,True)
  #if x>700:
  approx = cv2.approxPolyDP(c,epsilon,True)
  #print(len(approx))
  x,y,w,h = cv2.boundingRect(approx)


cv2.drawContours(img, [approx], 0, (0,255,255),2)
#corner_image4 = cv2.circle(corner_image4, (heig,width), radius=10, color=(0, 0, 255), thickness=-1)
cv2.imshow('Corners 4',img)


cv2.waitKey(0)



frame = np.copy(image)
Height, Width = 800 , 800
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners2_ret, corners2 = cv2.findChessboardCorners(gray, (Height, Width), None)

if corners2_ret == True:
	undist = cv2.undistort(gray, mtx, dist, None, newcamermtx)
	#undist= undist[y:y+h, x:x+w]
	corners2_ret, corners2 = cv2.findChessboardCorners(undist, (Height, Width), None) # points on the other plane 
	if corners2_ret:
		# compute perspective transform and apply to im
		h = cv2.findHomography(corners, corners2)[0]
		out = cv2.warpPerspective(im, h,(gray.shape[1], gray.shape[0]))
		# create mask and mask inverse 
		gray_out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)	
		ret, mask = cv2.threshold(gray_out, 10, 255, cv2.THRESH_BINARY)
		inv_mask = cv2.bitwise_not(mask)
		# create place for the warped image in the frame
		frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
		cv2.imshow('masked frame', frame)
		# grab only the ROI from the warped image
		out = cv2.bitwise_and(out, out, mask = mask)
		cv2.imshow('masked picture', out)
		# combine the two to create AR effect 
		frame = cv2.add(frame, out)
		cv2.imshow('warp', frame)
		cv2.waitKey(0)
		
	cv2.imshow('calib', frame)
	cv2.waitKey(0)









cv2.destroyAllWindows()
