import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):

	def __init__(self,batch_size,path,phase,max_side=320, classes=8, visual=False):
		'''Initialising the parameters'''
		self.batch_size=batch_size
		self.pointer =0
		self.path = path
		self.dataset = []
		self.phase=phase
		self.load_dataset()
		self.image_size = max_side
		self.classes = classes
		self.visual=visual
		self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

	def load_dataset(self):
		'''
		function : to read all images as well as the ground truth images from the directory 

		'''
		#print("loading data")
		for root, dirs, files in os.walk(os.path.join(self.path,'images',self.phase)):
		 	for file in files:
		 		if file.endswith(".jpg"):
		 			folder=root.split(os.sep)[-1]
		 			if not self.phase=='test':
		 				self.dataset.append((os.path.join(root, file),os.path.join(self.path,'annotation',self.phase,folder, file.split('_')[0]+'_label'+'.png')))
		 			else:
		 				self.dataset.append((os.path.join(root, file)))
		print(self.dataset)

	def encode(self,gt):
		'''
		function : convert the label to the one-hot encoding 
		parameters:
			gt : the ground truth with shape HxW with unique values of the class labels	
		
		return:
			ground_truth : the one hot encoded ground truth with shape n_classesxHxW
		'''
		
		ground_truth = np.zeros((self.classes, self.image_size,self.image_size),dtype='uint8')
		for index in range(self.classes-1):
			ground_truth[index,0:gt.shape[-2],0:gt.shape[-1]] = (gt==index)*1
		ground_truth[self.classes-1,0:gt.shape[-2],0:gt.shape[-1]]=(gt==255)*1

		return ground_truth

	def visualize(self,image, gt):
		'''
		Function : Visulise the image with different classes as separate images
		Parameters:
			image: the real image with dimenson HXW
			gt : the ground truth with shape HxW with unique values of the class labels
		Output:
			Display the image with no. of classes
		'''
		plt.imshow(image)
		plt.show()
		for index in range(self.classes):
			plt.imshow(gt[index,...])
			plt.show()

	def read_image_and_gt(self,data):
		file_name=data[0]

		image = cv2.imread(file_name)
		h,w,c = image.shape
		scale=h
		if w>h:
			scale=w
		scale=self.image_size/scale
		image = cv2.resize(image,(min(int(w*scale),self.image_size),min(int(h*scale),self.image_size)), interpolation=cv2.INTER_CUBIC)

		new_image = np.zeros((self.image_size,self.image_size,3 ),dtype='uint8')
		new_image[0:image.shape[-3],0:image.shape[-2],:]=image.copy()

		if not self.phase =='test':
			gt_name=data[1]
			gt = cv2.imread(gt_name)
			gt = cv2.resize(gt,(min(int(w*scale),self.image_size),min(int(h*scale),self.image_size)), interpolation=cv2.INTER_NEAREST)
			gt = self.encode(gt[:,:,0])
			if self.visual:
				self.visualize(new_image, gt)
			gt = gt.astype('float32')
			new_image = self.transform(new_image)
			return new_image, gt
		else:
			new_image = self.transform(new_image)
			return new_image


	def __getitem__(self, index):
		image_batch = torch.zeros(3,self.image_size,self.image_size).type(torch.FloatTensor)
		gt_batch = torch.zeros(self.classes,self.image_size,self.image_size).type(torch.uint8)

		if self.phase=='test':
			image_name = self.dataset[index]
			image = self.read_image_and_gt([image_name])
			return {'image': image, 'file_name':image_name}
		else:
			image_name, gt_name = self.dataset[index]
			image, gt = self.read_image_and_gt([image_name, gt_name] )
			return {'image': image, 'gt':gt, 'file_name':image_name}

	def __len__(self):
		return len(self.dataset)
