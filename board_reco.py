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

def contour(gray):
  canny = cv2.Canny(gray, 10,150)
  cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  points = 0
  Pmax = 0
  for c in cnts:
    epsilon = 0.1*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    if(len(approx)==4):
      points=approx
      L1 = np.hypot((points[0,0,0]-points[1,0,0]), (points[0,0,1]-points[1,0,1]))
      L2 = np.hypot((points[1,0,0]-points[2,0,0]), (points[1,0,1]-points[2,0,1]))
      L3 = np.hypot((points[2,0,0]-points[3,0,0]), (points[2,0,1]-points[3,0,1]))
      L4 = np.hypot((points[3,0,0]-points[0,0,0]), (points[3,0,1]-points[0,0,1]))
      P = L1+L2+L3+L4
      if(P>Pmax): Pmax = P
    #cv2.drawContours(image, [approx], 0, (0,255,0),2)
  #plt.imshow(image)
  #print(points)
  #print(Pmax)
  points = [[points[0],points[1],points[2],points[3]]]
  return points
  
def warp(points1, points2,img, img_resize):
	H,maks = cv2.findHomography(points1,points2)
	result = cv2.warpPerspective(img,H,img_resize)
	return result
	
	
import argparse


ap =  argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,help="Insert the path to input image")
args =  vars(ap.parse_args())
width, height = 800,800
image = cv2.imread(args["input"])
image = cv2.resize(image,(width,height)) # In order to resize the image 500x800

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image2 = np.copy(image)
points1 = contour(gray)
points2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
#result = warp(points1,points2,image,(width,height))
#cv2.imshow("result",result)
#cv2.waitKey(0)
#print(points1[0][1][0][0])
for i in range(4):
	
	image2 = cv2.circle(image2, (points1[0][i][0][0],points1[0][i][0][1]), 10, (0,0,255), 10)
	print(points1[0][i][0])
cv2.imshow("gray",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img_box = img.copy()
for i in range(8):
	for j in range(8):
		box1 = boxes[i,j]
		cv2.rectangle(img_box, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (255,0,0), 2)
		cv2.putText(img_box,"({},{})".format(i,j),(int(box1[2])-70, int(box1[3])-50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
		cv2.imshow("img",img_box)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

