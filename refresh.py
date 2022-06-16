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
threshold = 120
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
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 20, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])
clusters = cluster_points(intersected_points)
clusters = np.int0(clusters)

final_points = np.int0(clusters)
print("Final_points: "+str(len(final_points)))
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
"""
# Get row vector
rows = []
for k in range(len(final_points)):
	row = final_points[k,:,1]
	rows.append(int(sum(row)/len(row)))
print(rows)
"""
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


print(final_points)	

"""  
for j in range(len(final_points)):
  if(j == i1 or j == i2 or j == i3 or j == i4):
  	  cv2.circle( corner_image3, (final_points[j][0], final_points[j][1]), 3, (255,255,0), 1)
"""
# GRAFICAR
"""
graphic_points = final_points.copy()
graphic_points.shape = (-1,2)


for j in range(len(graphic_points)):
  cv2.circle(corner_image3, graphic_points[j], 5, (255,255,0), 3)
  cv2.imshow("Corners",corner_image3)
  cv2.waitKey(0)
"""

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

centers = get_centers(image, final_points)
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
for ki in range(5):
	MatChess = np.zeros((8,8))
	MatChess = MatChess.astype(int)
	cx = float(input("Introducir coordenada x: "))
	cy = float(input("Introducir coordenada y: "))
	r = detect_position(centers,cx,cy)
	print(r)
	MatChess[r//8][r%8] = 1
	print(MatChess)
	

def take_pictures(image, final_points):
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
			cv2.imshow("Border, circles",image3)
			cv2.waitKey(0)
			kimg = image3[iy-60:ey+10,ix-10:ex+10]
			cv2.imshow("Mini Images",kimg)
			#cv2.waitKey(0)
			#filename = str("board"+str(k)+str(j)+".jpeg")
			#print(filename)
			#path = "Mini"
			#cv2.imwrite(os.path.join(path,filename),kimg)
#take_pictures(image, final_points)

"""
#cv2.circle(corner_image3, final_points[7], 5, (255,255,0), 3)
#cv2.imshow("Corners",corner_image3)
#cv2.waitKey(0)
trp = final_points[i2]
tlp = final_points[i4]
brp = final_points[i1]
blp = final_points[i3]
"""

# homography
"""
rect = np.array(((tlp[0]+5, tlp[1]+5), (trp[0]-5, trp[1]+5), (brp[0]-5, brp[1]-5),
(blp[0]+5, blp[1]-5)), dtype="float32")
width = 200
height = 200
dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-
1]],dtype="float32")
M = cv2.getPerspectiveTransform(rect,dst)
warped_img = cv2.warpPerspective(image, M, (width, height))

#cv2.imshow("wi",warped_img)
#cv2.waitKey(0)
#warped_img1= cv2.copyMakeBorder(warped_img,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
#cv2.imshow("wib",warped_img1)
#cv2.waitKey(0)
#cv2.imwrite("Final.jpeg",warped_img1)


"""
 # Corner detection"""
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


corner_image4 = np.copy(image2)
cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4

cv2.drawContours(corner_image4, cnts, -1, (255,255,0), 2)

print('# Con: ', len(cnts))
for c in cnts:
  epsilon = 0.01*cv2.arcLength(c,True)
  #if x>700:
  approx = cv2.approxPolyDP(c,epsilon,True)
  #print(len(approx))
  x,y,w,h = cv2.boundingRect(approx)
  corner_image4 = cv2.circle(corner_image4, (x,y), radius=5, color=(0, 255, 0), thickness=-1)
  print("Sizes",w,"----",h)
  if(w>600 and h>600):
  	print('hi')
  	maxx = x
  	maxy = y
  	heig = h
  	width = w


cv2.drawContours(corner_image4, [approx], 0, (0,0,255),2)
#corner_image4 = cv2.circle(corner_image4, (heig,width), radius=10, color=(0, 0, 255), thickness=-1)
print(approx)
cv2.imshow('Corners 4',corner_image4)
cv2.waitKey(0)

"""
"""
corner_image5 = np.copy(image2)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

try: hierarchy = hierarchy[0]
except: hierarchy = []

height, width = edges.shape
min_x, min_y = width, height
max_x = max_y = 0

# computes the bounding box for the contour, and draws it on the frame,
for contour, hier in zip(contours, hierarchy):
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    if w > 80 and h > 80:
        cv2.rectangle(corner_image5, (x,y), (x+w,y+h), (255, 0, 0), 2)

if max_x - min_x > 0 and max_y - min_y > 0:
    cv2.rectangle(corner_image5, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
cv2.imshow('Corners 5',corner_image5)
cv2.waitKey(0)
"""
cv2.destroyAllWindows()

