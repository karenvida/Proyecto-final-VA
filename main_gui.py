import tkinter as tk
import PIL
from PIL import Image
from PIL import ImageTk
import imutils
import cv2
import numpy as np


from warp_image import lines_image
from warp_image import intersection_points
from warp_image import cluster_points
from warp_image import corners
from warp_image import reset
from segmentar_ord import crop_images
from comparision_f import per
from comparision_f import comp
from detection_pi import prediction
from refresh import convert
from refresh import get_centers
from refresh import detect_position
from chess.chess import chessboard
#from refresh import convert2image



#####################################################################
#####################################################################
#####################################################################


def digital(image):
	width, height = 800,800
	#background = cv2.imread("chess-board.png") 
	#pieces = cv2.imread("Pieces.png") 
	#background = cv2.resize(background,(width,height)) 
	#pieces = cv2.resize(pieces,(600,200)) # In order to resize the image 500x800
	image = cv2.resize(image,(width,height)) 
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
	#if(len(final_points)>81):
		#final_points = final_points[: len(final_points) - 9]  # Elimina la ultima fila de puntos
	"""Homography"""
	final_warp = corners(final_points,image)

	#cv2.imshow("Perpendicular image",final_warp)
	#cv2.waitKey(0)

	""" Cropping image"""
	centers = crop_images(final_points,image) # For keeping the cropped images it must be created a folder named "Mini" at your path
	for i in range(len(centers)):
		cv2.circle( corner_image, (centers[i][0], centers[i][1]), 3, (0,0,255), 1)
	#cv2.imshow("centroids",corner_image)
	#cv2.waitKey(0)
	#return(corner_image)
	""" Detecting movement changes"""
	board = reset()
	#print('New game!')
	#print(board)
	filename = str("to_rec.jpeg")
	cv2.imwrite(filename,image)
	
	pd=prediction(filename)
	#print('Detected pieces: ',len(pd))
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
	#cv2.imshow("centroids2",corner_image2)
	#cv2.waitKey(0)
	#New detection
	centers1 = get_centers(image, final_points)
	#MatChess = np.zeros((8,8))

	for i in range(len(pd)):
		
		#MatChess = MatChess.astype(int)
		cx = bases[i][0]
		cy = bases[i][1]
		r = detect_position(centers1,cx,cy)
		#print(r)
		board[r//8][r%8] =convert(pd[i][0])
	#return(corner_image2)
	#print(board)
	new_board = chessboard(board)
	return new_board
	#convert2image(board,background,pieces)
	#cv2.destroyAllWindows()

############################################################################
############################################################################
############################################################################


window = tk.Tk()
window.geometry("1600x950+100+10")
window.title("Chess detector")
window.resizable(width=False, height=False)
back = tk.PhotoImage(file="Back5.png")
back1 = tk.Label(window, image=back).place(x=0,y=0,relwidth=1,relheight=1)

title = tk.Label(window, bg="#1a646a", text="Real Time Chess Board", width=20, height=2, font=("Algerian",25,"bold"))
title.place(x=220,y=20)
subtitle1 = tk.Label(window, bg="#1a646a", text="Last board's picture", width=20, height=2, font=("Algerian",12,"bold"))
subtitle1.place(x=1000,y=20)
subtitle2 = tk.Label(window, bg="#1a646a", text="Digital board", width=20, height=2, font=("Algerian",12,"bold"))
subtitle2.place(x=1000,y=470)


et_video = tk.Label(window, bg="black")
et_video.place(x = 100, y = 160)
et_photo = tk.Label(window, bg="black")
et_photo.place(x = 1000, y = 90)
et_digital = tk.Label(window, bg="black")
et_digital.place(x = 1000, y = 540)

video = None


def take_photo():
	ret, frame = video.read()
	if ret == True:
		frame = imutils.resize(frame, width=450)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(frame)
		image = ImageTk.PhotoImage(image=img)
		et_photo.configure(image=image)
		et_photo.image = image
		pic = digital(frame)
		pic = imutils.resize(pic, width=400)
		#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
		img_pic = Image.fromarray(pic)
		image_pic = ImageTk.PhotoImage(image=img_pic)
		et_digital.configure(image=image_pic)
		et_digital.image = image_pic
	else:
		et_photo.image=""
		et_digital.image=""
		video.release()


def video_stream():
	global video
	video = cv2.VideoCapture("VideoChess2.mp4")
	iniciar()

def iniciar():
	global video
	ret, frame = video.read()
	if ret == True:
		frame = imutils.resize(frame, width=720)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(frame)
		image = ImageTk.PhotoImage(image=img)
		et_video.configure(image=image)
		et_video.image = image
		et_video.after(10,iniciar)
	else:
		et_video.image=""
		video.release()
		
def quitar():
	global video
	#et_video.place_forget()
	video.release()

boton1 = tk.Button(window, text="Capture Video Camera", bg="#5a4715", command=video_stream, font = ("Arial",20,"bold"), width=20, height=2)
boton1.place(x=100,y=730)
#boton2 = tk.Button(window, text="Stop Video Camera", bg="#5a4715", command=quitar, font = ("Arial",20,"bold"), width=20, height=2)
#boton2.place(x=100,y=820)
boton3 = tk.Button(window, text="Take picture", bg="#5a4715", command=take_photo, font = ("Arial",20,"bold"), width=15, height=2)
boton3.place(x=570,y=730)

window.mainloop()


