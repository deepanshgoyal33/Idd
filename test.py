import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import tiramisu
import utils.training as train_utils
from datagenerator_new import CustomDataset
from torch.autograd import Variable
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from utils.utils import load_model
from PIL import Image

# the path of the dataset 
path='./dataset/'

# the size of the mini batch
batch_size=1

# the writing path of the generated results
write_path = os.path.join(os.getcwd(),'result')

# the path of the trained model used for testing
weight_path='./weights/weights-500-0.238-0.238.pth'

def test_model(model, test_loader, viz=False):
	'''
	Testing the model
	parameters:
		model 		: the trained model 
		test_loader 	: the dataloder object used to fetch the test images 
	'''
	# used model in the evaluation mode
	model.eval()
	test_loss = 0
	test_error = 0
	with torch.no_grad():
		for idx, data_value in enumerate(test_loader):
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			data = Variable(data_value['image'].to(device), volatile=True)
			print("printing image paths")
			image_path = data_value['file_name'][0]
			print(image_path)
			output = model(data)
			print(output.shape)
			output = torch.argmax(output,dim=1).squeeze().cpu().detach().numpy()
			output = output[0:227,0:320]
			if viz==True:
				plt.imshow(output)
				plt.show()
				for index in range(8):
					plt.imshow(output[index,...])
					plt.show()
			background = (output==7)*255
			positive_class = (output<7)*1
			output1 = (output*positive_class)+(background*(1-positive_class))
			if viz==True:
				plt.imshow(positive_class)
				plt.show()
				plt.imshow(background)
				plt.show()
				plt.imshow(output1)
				plt.show()

			#output = [output1, output1, output1]
			#output = np.asarray(output)
			#print(output.shape,'.........................................')
			#output = output.transpose(1,2,0)
			folder = image_path.split(os.sep)[-2]
			print(folder)
			filename = image_path.split(os.sep)[-1].split('_')[0]+'_label.png'
			print(filename)

			# create the folder if not exist
			if not os.path.exists(os.path.join(write_path, folder)):
				os.makedirs(os.path.join(write_path, folder))
			
			cv2.imwrite(os.path.join(write_path,folder,filename), output)

def main():
	print('loading model:')
	model=load_model(weight_path)
	print('loading dataset:')
	test_loader = torch.utils.data.DataLoader(CustomDataset(batch_size,path,'test'),batch_size, shuffle=False, num_workers=1)
	print('number of samples',len(test_loader))
	test_model(model, test_loader)


if __name__=='__main__':
	main()
