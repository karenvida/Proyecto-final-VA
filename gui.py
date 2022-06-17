import tkinter as tk
import PIL
from PIL import Image
from PIL import ImageTk
import imutils
import cv2
import os
#from main_gui import main_pg

window = tk.Tk()
window.geometry("1600x950+100+10")
window.title("Chess detector")
window.resizable(width=False, height=False)
back = tk.PhotoImage(file="Back5.png")
back1 = tk.Label(window, image=back).place(x=0,y=0,relwidth=1,relheight=1)

title = tk.Label(window, bg="#1a646a", text="Digital Chess Board", width=20, height=2, font=("Algerian",25,"bold"))
title.place(x=600,y=20)

et_video = tk.Label(window, bg="black")
et_video.place(x = 100, y = 160)
et_photo = tk.Label(window, bg="black")
et_photo.place(x = 1000, y = 160)

video = None


def take_photo():
	ret, frame = video.read()
	if ret == True:
		frame = imutils.resize(frame, width=500)
		filename = str("Image_chess"+".jpeg")
		path = "Mini"
		cv2.imwrite(os.path.join(path,filename),frame)
		#new_board  = main_pg(frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		#img = Image.fromarray(frame)
		image = ImageTk.PhotoImage(image=img)
		et_photo.configure(image=image)
		et_photo.image = image
	else:
		et_photo.image=""
		video.release()


def video_stream():
	global video
	video = cv2.VideoCapture(0)
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
boton2 = tk.Button(window, text="Stop Video Camera", bg="#5a4715", command=quitar, font = ("Arial",20,"bold"), width=20, height=2)
boton2.place(x=100,y=820)
boton3 = tk.Button(window, text="Take picture", bg="#5a4715", command=take_photo, font = ("Arial",20,"bold"), width=15, height=2)
boton3.place(x=570,y=730)

window.mainloop()


