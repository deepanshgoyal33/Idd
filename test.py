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

path='./dataset/'
batch_size=1
write_path = './result'
weight_path='./weights/weights-500-0.238-0.238.pth'

def load_model(path=None):
	model = tiramisu.FCDenseNet103(n_classes=8).cuda()
	weights = torch.load(path)
	model.load_state_dict(weights['state_dict'])
	return model

def test_model(model, test_loader):
	model.eval()
	test_loss = 0
	test_error = 0
	with torch.no_grad():
		for idx, data_value in enumerate(test_loader):
			data = Variable(data_value['image'].cuda(), volatile=True)
			image_path = data_value['file_name'][0]
			print(image_path)
			output = model(data)
			print(output.shape)
			output = torch.argmax(output,dim=1).squeeze().cpu().detach().numpy()
			output = output[0:227,0:320]
#			plt.imshow(output)
#			plt.show()
#			for index in range(8):
#				plt.imshow(output[index,...])
#				plt.show()
#			print(np.unique(output))
			background = (output==7)*255
			positive_class = (output<7)*1
	#		plt.imshow(positive_class)
	#		plt.show()
	#		plt.imshow(background)
	#		plt.show()
			output1 = (output*positive_class)+(background*(1-positive_class))
			#plt.imshow(output1)
			plt.show()

			#output = [output1, output1, output1]
			#output = np.asarray(output)
			print(output.shape,'.........................................')
			#output = output.transpose(1,2,0)
			folder = image_path.split(os.sep)[-2]
			filename = image_path.split(os.sep)[-1].split('_')[0]+'_label.png'
			if not os.path.exists(os.path.join(write_path, folder)):
				os.mkdir(os.path.join(write_path, folder))
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
