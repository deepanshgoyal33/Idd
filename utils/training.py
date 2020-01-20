import os
import sys
import math
import string
import random
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = './weights/'

def save_weights(model, epoch, loss, err):
    '''
    Function - Saving the weights of while training
    Parameters- Epoch number,loss and error(all are of int type)

    '''
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)

    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)

    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    '''dangling
    '''
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    '''
    Converting the the tensor generated, to the image with the class-number of the highest probability
    Parameters- 
        output_batch- tensor of size `Batch_size` X `No_of_classes` X `H` X `W`
    Output- 
        Indices with dimension `Batch_size` X `H` X `W` with individual pixel having the class predicted
    '''
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices


def criterion(probas,true_1_hot,eps=1e-7):
    '''
    Customiszed Loss Function
    Parameters-
        probas-the generated output from the model
        true_1_hot- The one hot encoded grounf truth
    Output-
        Loss
    '''
    intersection=torch.sum(probas*true_1_hot,0)
    intersection=torch.sum(intersection,-1)
    intersection=torch.sum(intersection,-1)
    cardinality = torch.sum(probas + true_1_hot,0)
    cardinality = torch.sum(cardinality,-1)
    cardinality = torch.sum(cardinality,-1)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    loss = (1 - dice_loss)    
    return loss	

def train(model, trn_loader, optimizer, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    counter=0;
    for idx, data in enumerate(trn_loader):
        inputs = Variable(data['image'].cuda())
        targets = Variable(data['gt'].cuda())
        optimizer.zero_grad()
        output = model(inputs)
        loss=[]
        for index in range(8):
            temp= criterion(output[:,index,0:227,0:320],targets[:,index,0:227,0:320])
            loss.append(temp)
        loss=torch.mean(torch.stack(loss))
        loss.backward()
        optimizer.step()
        counter=counter+1
        trn_loss += loss.data
        if counter%100==0:
            print('Epoch {:d} \t Iteration {:d} \t Train - Loss: {:.4f}'.format(epoch, idx+1, trn_loss/max(counter,1))) 
    trn_loss /= len(trn_loader)
    return trn_loss

def validation(model, test_loader, epoch=1):
    model.eval()
    test_loss = 0
    for idx, data in enumerate(test_loader):
        print(data.keys())
        print(data['gt'])
        print(data['file_name'])
        data = Variable(data['image'].cuda())
        target = Variable(data['gt'].cuda())

        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    print('VALIDATION : ',test_loss.data)
    return test_loss

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    '''Initailisng the weights'''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

