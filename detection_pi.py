

#sys.path.append(".")
#import cv2
import torch
from torch import nn
import torch.nn.functional as F
import sys
from PIL import Image
from funct_det import Yolov4
from tool.utils import *

def prediction(imgfile):
	n_classes = 13
	weightfile = 'Yolov4_epoch78.pth'
	namesfile = '_classes.txt'
	model = Yolov4(n_classes=n_classes)

	pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
	model.load_state_dict(pretrained_dict)


	use_cuda = 1
	if use_cuda:
		model.cuda()
	#img = cv2.cvtColor(imgfile, cv2.COLOR_BGR2RGB)
	img = Image.open(imgfile).convert('RGB')
	sized = img.resize((800, 800))

	boxes = do_detect(model, sized, 0.5, n_classes,0.4, use_cuda)

	class_names = load_class_names(namesfile)
	data = plot_boxes(sized, boxes, 'predictions.jpg', class_names)
	#print(image)
	return data
