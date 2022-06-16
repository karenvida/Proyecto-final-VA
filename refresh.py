# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
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


# Take mini pictures
def get_centers(image, final_points):
	centers = np.zeros((64,2))
	index = 0
	for k in range(1,len(final_points)):
		for j in range(1,9):
			image3 = image.copy()
			Fcoord = np.array([final_points[k-1][j-1], final_points[k-1][j], final_points[k][j-1], final_points[k][j] ])
			ix = min(Fcoord[:,0])
			ex = max(Fcoord[:,0])
			iy = min(Fcoord[:,1])
			ey = max(Fcoord[:,1])
			cx = (ix+ex)/2
			cy = (iy+ey)/2
			centers[index][0]=cx
			centers[index][1]=cy
			index = index+1
	return centers

#centers = get_centers(image, final_points)
#print(centers)


# MATRIZ DEL TABLERO

def detect_position(centers,cx,cy):
	dist_min = 1000000
	u = None
	for i in range(64):
		a = np.array((cx,cy))
		b = np.array((centers[i][0],centers[i][1]))
		dist = np.linalg.norm(b-a)
		if(dist < dist_min):
			dist_min = dist
			u = i
	return u
#PRUEBAS MATRIZ
	"""
for ki in range(5):
	MatChess = np.zeros((8,8))
	MatChess = MatChess.astype(int)
	cx = float(input("Introducir coordenada x: "))
	cy = float(input("Introducir coordenada y: "))
	r = detect_position(centers,cx,cy)
	print(r)
	MatChess[r//8][r%8] = 1
	print(MatChess)
	"""
def convert(nombre):
	if(nombre == 'white-queen'):
		return 'Q'
	if(nombre == 'white-king'):
		return 'K'
	if(nombre == 'white-pawn'):
		return 'P'
	if(nombre == 'white-rook'):
		return 'R'
	if(nombre == 'white-bishop'):
		return 'B'
	if(nombre == 'white-knight'):
		return 'N'
	if(nombre == 'black-queen'):
		return 'q'
	if(nombre == 'black-king'):
		return 'k'
	if(nombre == 'black-pawn'):
		return 'p'
	if(nombre == 'black-rook'):
		return 'r'
	if(nombre == 'black-bishop'):
		return 'b'
	if(nombre == 'black-knight'):
		return 'n'
def convert2image(matrix,board_image,pieces):
	#print(len(matrix))
	#print(board_image.shape())
	print(len(pieces[0]))
	new_image = cv2.add(board_image[0:600][0:200],pieces)
	cv2.imshow("centroids3",new_image)
	cv2.waitKey(0)
	#for i in range(8):
	#	for j in range(8):
	#		print(board_image.shape[0],',,,',board_image.shape[1])
		
	
cv2.destroyAllWindows()

