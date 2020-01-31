import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import tiramisu
import utils.training as train_utils
from datagenerator_new import CustomDataset
from utils.utils import load_model
path='./dataset/'
batch_size=1
N_EPOCHS=500

LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
torch.cuda.manual_seed(0)

print('loading model:')
model=load_model()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)

print('loading Training Dataset:')
train_loader = torch.utils.data.DataLoader(CustomDataset(batch_size,path,'train'),batch_size, shuffle=True, num_workers=4)
print('loading Validation Dataset:')
valid_loader = torch.utils.data.DataLoader(CustomDataset(batch_size,path,phase='val'),batch_size, shuffle=False, num_workers=4)
print('number of iteration per epoch',len(train_loader))

for epoch in range(1, N_EPOCHS+1):
	since = time.time()
	train_utils.validation(model, valid_loader)
	### Train ###
	print('TRAINING ---->')
	trn_loss = train_utils.train(model, train_loader, optimizer, epoch)
	print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))
	time_elapsed = time.time() - since
	print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	### Checkpoint ###
	if epoch%1==0:
		print('VALIDATION ---->')		
		train_utils.validation(model, valid_loader)
		print('SAVING WEIGHTS ---->')		
		train_utils.save_weights(model, epoch, trn_loss, trn_loss)

	### Adjust Lr ###
	train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)
