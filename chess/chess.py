import cv2
import numpy as np

def chessboard(matriz):
	dim = 480
	size = int(dim/8)




	background = cv2.imread("chess/chessboard.jpg") 
	background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
	background = cv2.resize(background,(dim,dim))

	bp = cv2.imread("chess/bP.jpg") 
	bp = cv2.cvtColor(bp, cv2.COLOR_BGR2GRAY)
	bp = cv2.resize(bp,(size,size))

	bk = cv2.imread("chess/bK.jpg") 
	bk = cv2.cvtColor(bk, cv2.COLOR_BGR2GRAY)
	bk = cv2.resize(bk,(size,size))

	bb = cv2.imread("chess/bB.jpg") 
	bb = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
	bb = cv2.resize(bb,(size,size))

	bn = cv2.imread("chess/bN.jpg") 
	bn = cv2.cvtColor(bn, cv2.COLOR_BGR2GRAY)
	bn = cv2.resize(bn,(size,size))

	bq = cv2.imread("chess/bQ.jpg") 
	bq = cv2.cvtColor(bq, cv2.COLOR_BGR2GRAY)
	bq = cv2.resize(bq,(size,size))

	br = cv2.imread("chess/bR.jpg") 
	br = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
	br = cv2.resize(br,(size,size))

	wp = cv2.imread("chess/wP.jpg") 
	wp = cv2.cvtColor(wp, cv2.COLOR_BGR2GRAY)
	wp = cv2.resize(wp,(size,size))

	wk = cv2.imread("chess/wK.jpg") 
	wk = cv2.cvtColor(wk, cv2.COLOR_BGR2GRAY)
	wk = cv2.resize(wk,(size,size))

	wb = cv2.imread("chess/wB.jpg") 
	wb = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
	wb = cv2.resize(wb,(size,size))

	wn = cv2.imread("chess/wN.jpg") 
	wn = cv2.cvtColor(wn, cv2.COLOR_BGR2GRAY)
	wn = cv2.resize(wn,(size,size))

	wq = cv2.imread("chess/wQ.jpg") 
	wq = cv2.cvtColor(wq, cv2.COLOR_BGR2GRAY)
	wq = cv2.resize(wq,(size,size))

	wr = cv2.imread("chess/wR.jpg") 
	wr = cv2.cvtColor(wr, cv2.COLOR_BGR2GRAY)
	wr = cv2.resize(wr,(size,size))

	def replace(cut, img):
		aux = cut[0][0]
		num = img[0][0]
		cut = np.copy(img)
		cut[cut==num] = aux
		return cut
		
	def select(i, j, img):
		cut = background[size*i:size*(i+1), size*j:size*(j+1)]
		background[size*i:size*(i+1), size*j:size*(j+1)] = replace(cut,img)

	for i in range(8):
		for j in range(8):
			if(matriz[i][j] != '.'):
				if(matriz[i][j] == 'R'):
					select(i,j,wr)
				elif(matriz[i][j] == 'N'):
					select(i,j,wn)
				elif(matriz[i][j] == 'B'):
					select(i,j,wb)
				elif(matriz[i][j] == 'Q'):
					select(i,j,wq)
				elif(matriz[i][j] == 'K'):
					select(i,j,wk)
				elif(matriz[i][j] == 'P'):
					select(i,j,wp)
				elif(matriz[i][j] == 'r'):
					select(i,j,br)
				elif(matriz[i][j] == 'n'):
					select(i,j,bn)
				elif(matriz[i][j] == 'b'):
					select(i,j,bb)
				elif(matriz[i][j] == 'q'):
					select(i,j,bq)
				elif(matriz[i][j] == 'k'):
					select(i,j,bk)
				elif(matriz[i][j] == 'p'):
					select(i,j,bp)
	#cv2.imshow("chessboard", background)
	#cv2.waitKey(0)
	return background
	
matriz = [['R','N','B','Q','K','B','N','R'],
	['P','P','P','P','P','P','P','P'],
	['.','.','.','.','.','.','.','.'],
	['.','.','.','.','.','.','.','.'],
	['.','.','.','.','.','.','.','.'],
	['.','.','.','.','.','.','.','.'],
	['p','p','p','p','p','p','p','p'],
	['r','n','b','q','k','b','n','r']]
	
#chessboard(matriz)	
