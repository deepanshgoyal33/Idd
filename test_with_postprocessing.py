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
#from utils.crf import dense_crf
from torchvision import transforms
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

path='./dataset/'
batch_size=1
write_path = './result_posprocessing'
weight_path='./weights/weights-500-0.238-0.238.pth'


def load_model(path=None):
	'''
		Function to load model
	'''
	model = tiramisu.FCDenseNet103(n_classes=8).cuda()
	weights = torch.load(path)
	model.load_state_dict(weights['state_dict'])
	return model



def postprocessing(labels,image_path):
	'''
	Fuction: To return the psotprocessed output ie. using the dense crf funtion for incresing the sharpness of the edges
		Parameters:
			Labels: array of size N-classesXHXW containg the probabilities of each pixel and of each class
			image_path: Path to the image(str)
		Outputs:
			result: post precessed output

	'''
	img = cv2.imread(image_path)

	print('cross check',np.unique(labels))
	labelsx = np.argmax(labels,axis=0)
	print(labelsx)

	unique = np.unique(labelsx)
	n_labels = len(unique)

	potentials =[]

	for index in range(len(unique)):
		print(unique[index], labels[unique[index],...].shape)
		potentials.append(labels[unique[index],...])

	potentials = np.stack(potentials)

	print('mare hai', potentials.shape)

	labels = potentials
	labels = -np.log(labels)
	d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
	U = np.ascontiguousarray(labels.reshape(n_labels, img.shape[1]* img.shape[0]))
	#U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=15)
	print('lelo',U.shape)
	d.setUnaryEnergy(U)


	# This creates the color-independent features and then add them to the CRF
	feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
	d.addPairwiseEnergy(feats, compat=5, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features and then add them to the CRF
	feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=img, chdim=2)
	d.addPairwiseEnergy(feats, compat=5,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)

#	# This adds the color-independent term, features are the locations only.
#	d.addPairwiseGaussian(sxy=(20, 20), compat=3, kernel=dcrf.DIAG_KERNEL,
#                          normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#	# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
#	d.addPairwiseBilateral(sxy=(80, 80), srgb=(1,1,1), rgbim=img,
#                           compat=10,
#                           kernel=dcrf.DIAG_KERNEL,
#                           normalization=dcrf.NORMALIZE_SYMMETRIC)

	# Run five inference steps.
	Q = d.inference(50)
#	print('hhhhhhhheeeeeeeeellloooooooooooooo',Q.shape)	
	MAP = np.argmax(Q, axis=0)
	print('hhhhhhhheeeeeeeeellloooooooooooooo',MAP.shape)
	MAP = MAP.reshape(img.shape[0],img.shape[1])
	uniquex = np.unique(MAP)
	result = np.zeros((8,MAP.shape[0],MAP.shape[1]),dtype='uint8')
	print(unique)
	print(np.unique(MAP))
	for index in range(len(uniquex)):
		result[unique[index],...]=(MAP==uniquex[index])
	result = np.argmax(result,axis=0)

	return result

def visualization(image_orig,output_prev, output ):
	plt.subplot(2,1,1)
	plt.imshow(image_orig)
	plt.subplot(2,1,2)
	plt.imshow(np.concatenate((output_prev, output), axis=1))
	plt.show()

def test_model(model, test_loader,use_dense_crf=False, viz=True):
	model.eval()
	test_loss = 0
	test_error = 0
	with torch.no_grad():
		for idx, data_value in enumerate(test_loader):
			data = Variable(data_value['image'].cuda(), volatile=True)

			image_path = data_value['file_name'][0]
			image_orig = cv2.imread(image_path)
			print(image_path)
			output = model(data)
			print(output.shape)
			output_prev = output[:,:,0:227,0:320].squeeze()
			if use_dense_crf:
				output = postprocessing(output_prev, image_path)
			else:
				output = output_prev
			output_prev = torch.argmax(output_prev,dim=0).squeeze().cpu().detach().numpy()
			print(output_prev.shape, output.shape)
			if viz:
				visualization(image_orig,output_prev, output )
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
			#plt.show()

			#output = [output1, output1, output1]
			#output = np.asarray(output)
			print(output1.shape,'.........................................')
			#output = output.transpose(1,2,0)
			folder = image_path.split(os.sep)[-2]
			filename = image_path.split(os.sep)[-1].split('_')[0]+'_label.png'
			if not os.path.exists(os.path.join(write_path, folder)):
				os.mkdir(os.path.join(write_path, folder))
			cv2.imwrite(os.path.join(write_path,folder,filename), output1)

def main():
	print('loading model:')
	model=load_model(weight_path)
	print('loading dataset:')
	test_loader = torch.utils.data.DataLoader(CustomDataset(batch_size,path,'test'),batch_size, shuffle=False, num_workers=1)
	print('number of samples',len(test_loader))
	test_model(model, test_loader,use_dense_crf=True,viz=False)


if __name__=='__main__':
	main()
