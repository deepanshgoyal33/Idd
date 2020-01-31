from models import tiramisu

def load_model(path=None):
	'''
	function to load the model 
	parameters :
		path : the path of the model state dictionary 
	return:
		model: the model along with the random weights if path is none otherwise initalized with pretrained weight 
	'''
	model = tiramisu.FCDenseNet103(n_classes=8).cuda()
	if path is not None:
		weights = torch.load(path)
		model.load_state_dict(weights['state_dict'])
	return model


