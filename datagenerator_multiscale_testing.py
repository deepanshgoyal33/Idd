import numpy as np
#import cv2
#import os
#import torch
#from utils import *
#from matplotlib import pyplot as plt 

import torch
import cv2
import os
#from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

#import random
#import sys


class CustomDataset(Dataset):
	"""
	A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
	"""
	def __init__(self,batch_size,path,phase,scale,max_side=320, classes=8, visual=True):
		self.batch_size=batch_size
		self.pointer =0
		self.path = path
		self.dataset = []
		self.phase=phase
		self.load_dataset()
		self.image_size = max_side
		self.classes = classes
		self.visual=visual
		self.scale=scale
		self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

	def load_dataset(self):
		'''
		function : to read all images as well as the ground truth images from the directory 
		'''
		for root, dirs, files in os.walk(os.path.join(self.path,'images',self.phase)):
			for file in files:
				if file.endswith(".jpg"):
					folder=root.split(os.sep)[-1]
					if not self.phase=='test':
						self.dataset.append((os.path.join(root, file),os.path.join(self.path,'annotation',self.phase,folder, file.split('_')[0]+'_label'+'.png')))
					else:
						self.dataset.append((os.path.join(root, file)))

	def decode(self,gt, scale):
		'''
		Function- Returning the ground truth with the new scaled dimensions
		Parameters-
			gt- Ground truth of size HXW with unique labels of class labels
			scale - scaling factor 
		Outputs-
			ground_truth : the one hot encoded ground truth with shape [n_classes]x[(scale_factor)*H]x[(scale_factor)*W]

		'''
		desired_size=int(scale*self.image_size)
		new_h = int(scale*h)
		new_w = int(scale*w)
		ground_truth = np.zeros((self.classes, desired_size,desired_size),dtype='uint8')		
		for index in range(self.classes-1):
			ground_truth[index,0:new_h,0:new_w] = (gt==index)*1
		ground_truth[self.classes-1,0:new_h,0:new_w]=(gt==255)*1
		return ground_truth

	def visualize(self,image, gt):
		'''
		Function- to visulise the images and ground truths
		'''
		plt.imshow(image)
		plt.show()
		for index in range(self.classes):
			plt.imshow(gt[index,...])
			plt.show()
	
	def read_image_and_gt(self,data):
		''' 
			Function: To read the images and the ground truths and scale it to the desired value
			Parameters: 
				data - It is in tuple form with zeroth index as the location of the image and the other index as the location of the segmented ground truth
			Output-
				image_various_scale - return the image in new scale.
		''' 
		file_name=data[0]
		image = cv2.imread(file_name)
		h,w,c = image.shape
		image_various_scale=[]
		gt_various_scale=[]
		for index in range(len(self.scale)):
			desired_size=int(self.scale[index]*self.image_size)
			new_h = int(self.scale[index]*h)
			new_w = int(self.scale[index]*w)
			resized_image = cv2.resize(image,(new_w, new_h),interpolation=cv2.INTER_CUBIC)
			new_image = np.zeros((desired_size,desired_size,3 ),dtype='uint8')
			new_image[0:new_h,0:new_w,:]=resized_image.copy()

			if self.phase =='valid':
				gt_name=data[1]
				gt = cv2.imread(gt_name)
				resized_gt = cv2.resize(gt,(new_w, new_h),interpolation=cv2.INTER_INTER)
				resized_gt = self.decode(resized_gt[:,:,0], self.scale[index])
				if self.visual:
					self.visualize(new_image, resized_gt)
				resized_gt = resized_gt.astype('float32')
				new_image = self.transform(new_image)
				gt_various_scale.append(resized_gt)
				image_various_scale.append(new_image)
			else:
				print('testing --------------------')
				new_image = self.transform(new_image)
				image_various_scale.append(new_image)
		if self.phase=='valid':
			return image_various_scale, gt_various_scale
		else:
			return image_various_scale

	def __getitem__(self, index):
		image_batch = torch.zeros(3,self.image_size,self.image_size).type(torch.FloatTensor)
		gt_batch = torch.zeros(self.classes,self.image_size,self.image_size).type(torch.uint8)
		
		if self.phase=='test':
			image_name = self.dataset[index]
			print('testing --------------------')
			image = self.read_image_and_gt([image_name])
			datadict={}
			for index in range(len(self.scale)):
				print('image_scale_'+str(index))
				datadict['image_scale_'+str(index)]=image[index]
			datadict['file_name']=image_name
			return datadict

		else:
			image_name, gt_name = self.dataset[index]
			print('hello------------------------')
			image, gt = self.read_image_and_gt([image_name, gt_name] )
			datadict={}
			for index in range(len(self.scale)):
				print('image_scale_'+str(index))
				datadict['image_scale_'+str(index)]=image[index]
			for index in range(len(self.scale)):
				datadict['gt_scale_'+str(index)]=gt[index]
			datadict['file_name']=image_name
			return datadict
		
	def __len__(self):
		return len(self.dataset)




