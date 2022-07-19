import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import seaborn as sn
import pandas as pd
from PIL import Image, ImageDraw
from io import BytesIO
import json
import torch.utils.data as data
from torchvision import models
from torch import nn
from torch import optim
from tqdm import tqdm
import torchvision.datasets as dset
from torch.autograd import Variable
from model import CNN
from loss import loss_coteaching
import argparse
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from collections import Counter 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from utils import *
from randaugment import RandAugment, ImageNetPolicy
from sklearn.preprocessing import MinMaxScaler
from loss import *
from dataset import*
parser = argparse.ArgumentParser()

parser.add_argument('--cfg', type=str, default='./noise_longtail/config/0.1_red.yaml')
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--device_ids', type=list, default=[0])
parser.add_argument('--save_model',type = int,default=0)
parser.add_argument('--freeze',type = int,default=0)
parser.add_argument('--iterative', type=int, default=0)
args = parser.parse_args()

seed=123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load config
args = update_args(args)

# load path


TRAIN_DATA_PATH = args.train_path
TEST_DATA_PATH = args.test_path
RESAMPLE_PATH = args.resample_path
if args.iterative ==1 or args.iterative ==2:
    pth_file_stage_1= args.iter_pth_stage_1_path
    pth_file_stage_2= args.iter_pth_stage_2_path
    FIANL_WEIGHT_PATH = args.iter_final_weight_path
    FINAL_MODEL_PATH = args.iter_final_model_path
    CLASSIFER_PATH = args.iter_classifer_path
    JSON_PATH = args.iter_json_path
else:

    pth_file_stage_1= args.pth_stage_1_path
    pth_file_stage_2= args.pth_stage_2_path
    FINAL_MODEL_PATH = args.final_model_path
    FIANL_WEIGHT_PATH = args.final_weight_path
    CLASSIFER_PATH = args.classifer_path
    JSON_PATH = args.json_path



def compute_adjustment(train_loader, tro):
    """compute the base probabilities"""

    label_freq = {}
    for i, (img,target,_) in tqdm(enumerate(train_loader)):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments


def train_epoch(net, loss_function, dataloader, optimizer,args):
  train_loss = train_acc = correct = 0
  net.train()
  for i ,(images,labels,paths) in enumerate(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    # get instance weight
    weight_list = []
    for path in paths:
      weight_list.append(args.weight_dict[path])
    list_sum = np.sum(weight_list)
    weight_list = torch.tensor( [i/list_sum for i in weight_list])
    optimizer.zero_grad()
    outputs = net(images)
    loss = torch.sum(weight_list.to(device)*loss_function(outputs, labels))
    train_loss+= loss
    loss_r = 0
    for parameter in model.parameters():
        loss_r += torch.sum(parameter ** 2)
    loss = loss + args.weight_decay * loss_r

    loss.backward()
    
    optimizer.step()

    _, pred = torch.max(outputs, 1)  
    correct += pred.eq(labels).cpu().sum().item() 
        
  train_loss/=len(dataloader)
  train_acc=100.*correct/len(dataloader.dataset)
  return train_loss, train_acc



#@title test method
def test(net, loss_function, loader,args):
  net.eval()
  test_loss = 0
  correct = 0
  
  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = net(data)
      if (args.flag) :
          output = output - args.adjustment
      test_loss += loss_function(output, target)
      _, pred = torch.max(output, 1)  
      correct += pred.eq(target).cpu().sum().item() 
  acc = 100.*correct/len(loader.dataset)
  test_loss /= len(loader)
  
  return acc, test_loss

def train_test_model(train_loader,test_loader,loss_function,model,args):
  if (args.save_model):
    print( "this model will be saved !")
  args.flag = 0
  args.adjustment = compute_adjustment(train_loader, 1.0)
  net = model.cuda()
 #freeze
  if args.freeze :
      for param_name, param in net.module.named_parameters():
          if 'fc' not in param_name and 'layer4.1' not in param_name:
          #if 'fc' not in param_name  and 'layer4' not in param_name:
              param.requires_grad = False
      optimizer = optim.SGD([p for p in net.module.parameters() if p.requires_grad], lr=args.lr, momentum=0.9,weight_decay = args.weight_decay)  
  else:
      optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,weight_decay = args.weight_decay)  
  #scheduler = StepLR(optimizer,step_size = 5 ,gamma = 0.5)
  scheduler = CosineAnnealingLR(optimizer,T_max = args.num_epochs,eta_min=0,last_epoch=-1,verbose=False)
  logs={'train_loss':[],'train_acc':[], 'test_loss':[], 'test_acc':[],'eval_loss':[]}
  # observe the performance in stage 1
  # add de-longtail
  base_acc_test, loss_test=test(net, loss_function, test_loader,args)
  max_acc_test = base_acc_test
  print('--------base test acc in stage 1:',base_acc_test,'-------------')

  # select LA as the LT meothods
  args.flag = 1
  for epoch in tqdm(range(args.num_epochs)):
    loss_function = nn.CrossEntropyLoss(reduction = 'none')
    #loss_function = BlSoftmaxLoss(train_loader)  # select bsmax as LT methods
    train_loss, train_acc = train_epoch(net, loss_function , train_loader, optimizer,args)
    
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    ce_loss = nn.CrossEntropyLoss()
    acc_test, loss_test=test(net,ce_loss, test_loader,args)
    is_best = True if acc_test > max_acc_test else False
    if is_best:
      max_acc_test = acc_test
      print('=> find better final_model ...')
      if args.save_model :
        save_checkpoint(
                      {
                          'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          
                      },
                      checkpoint =FINAL_MODEL_PATH,
                      is_best=is_best)

    scheduler.step()
    logs['test_loss'].append(loss_test)
    logs['test_acc'].append(acc_test)

    print('Epoch [{}/{}],\ttrain loss:{:.4f},\ttrain acc:{:.2f},\ttest loss:{:.4f},\ttest accuracy:{:.2f},\t lr:{:.4f}'.format(epoch,args.num_epochs,train_loss,train_acc, loss_test, acc_test,optimizer.param_groups[0]['lr']))
  return logs,base_acc_test,max_acc_test



#load data

train_transform = transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_data = imagenet_with_path(TRAIN_DATA_PATH, transform=train_transform)
test_data =  dset.ImageFolder(TEST_DATA_PATH, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                              batch_size = args.batch_size,
                                              shuffle = True,num_workers=4,pin_memory=True)


test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = args.batch_size, 
                                      shuffle = False,num_workers=4)

# load model & dict
model = models.resnet18(pretrained=False) 
model.fc= nn.Linear(in_features=model.fc.in_features, out_features=args.cls_num)
resnet_model=nn.DataParallel(model,device_ids=[0,1,2,3])
model_dict= torch.load(pth_file_stage_1)
resnet_model.load_state_dict(model_dict['state_dict'])


with open(FIANL_WEIGHT_PATH,'r') as load_f:
    weight_dict = json.load(load_f)

args.weight_dict = weight_dict


loss_function =nn.CrossEntropyLoss()
logs_baseline,base_acc_test,max_acc_test = train_test_model(train_loader,test_loader,loss_function,resnet_model,args)

print(TRAIN_DATA_PATH,'\n')
print('---------the best test  acc in stage 1:',base_acc_test,'----------')
print('---------the best test acc in stage 2:',max_acc_test,'----------')




