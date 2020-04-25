import torch
from models import tiramisu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(path=None):
	'''
	function to load the model 
	parameters :
		path : the path of the model state dictionary 
	return:
		model: the model along with the random weights if path is none otherwise initalized with pretrained weight 
	'''
	model = tiramisu.FCDenseNet103(n_classes=8).to(device)
	if path is not None:
		weights = torch.load(path, map_location = device)
		model.load_state_dict(weights['state_dict'])
	return model


