import os
from torch.nn.modules.activation import ReLU
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
import json
import torch.utils.data as data
from torchvision import models
from torch import nn
from torch import optim
from tqdm import tqdm
import torchvision.datasets as dset
from torch.autograd import Variable
from model import CNN
from loss import *
from sklearn.preprocessing import MinMaxScaler
import argparse
import importlib
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from collections import Counter 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from utils import *
from randaugment import RandAugment, ImageNetPolicy
from model import *
from dataset import *

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

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_epoch(net, loss_function, dataloader, optimizer,args):
  train_loss = train_acc = correct = 0
  net.train()
  for i ,(images,labels,paths) in enumerate(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    if (args.mix_up):
      inputs, targets_a, targets_b, lam = mixup_data(images, labels, 1.)
      inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
      outputs = net(inputs)
      loss_func = mixup_criterion(targets_a, targets_b, lam)
      loss = loss_func(loss_function, outputs)
      train_loss+= loss
    elif (args.mix_up_weight):
      score_list = []
      for j in range(images.size(0)):
        temp_label = str(labels[j].cpu().numpy())
        if len(temp_label) == 1:
         temp_label = '0' +temp_label
        score_list.append(args.score_dict[temp_label][paths[j]])
      mixed_x, y_a, y_b, score_list,new_score_list = mixup_data_weight(images, labels,score_list)
      outputs = net(mixed_x)
      ce_loss = nn.CrossEntropyLoss(reduction = 'none')
      loss = torch.mean(score_list.to(device)*ce_loss(outputs,y_a) + new_score_list.to(device)*ce_loss(outputs,y_b))
      train_loss+= loss

    else:    
      outputs = net(images)
      loss = loss_function(outputs, labels)
      train_loss+= loss
    #regularization
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

def test(net, loss_function, loader,args):
  net.eval()
  test_loss = 0
  correct = 0
  
  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = net(data)
      test_loss += loss_function(output, target)
      _, pred = torch.max(output, 1)  
      correct += pred.eq(target).cpu().sum().item() 
  acc = 100.*correct/len(loader.dataset)
  test_loss /= len(loader)
  
  return acc, test_loss


def train_test_model(train_loader,test_loader,loss_function,model,args):
  if (args.save_model):
    print( "this model will be saved !")
  if (args.resample):
    print("-----------resample--------------\n")
  if (args.mix_up):
    print("-------------mix_up--------------\n")
  

  if (args.mix_up_weight):
    print("-------------mix_up_weight--------------\n")


  net = model.cuda()
  optimizer_1 = optim.SGD(resnet_model.parameters(), lr=args.lr, momentum=0.9,weight_decay = args.weight_decay)  
  #scheduler = StepLR(optimizer_1,step_size = 50 ,gamma = 0.1)
  scheduler = CosineAnnealingLR(optimizer_1,T_max = args.num_epochs,eta_min=0,last_epoch=-1,verbose=False)
  logs={'train_loss':[],'train_acc':[], 'test_loss':[], 'test_acc':[],'eval_loss':[]}

  max_acc_test = 0
  for epoch in tqdm(range(args.num_epochs)):
    train_loss, train_acc = train_epoch(net, loss_function, train_loader, optimizer_1,args)
    
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)

    acc_test, loss_test=test(net, loss_function, test_loader,args)
    is_best = True if acc_test > max_acc_test else False
    if is_best:
      max_acc_test = acc_test
      print('=> find better checkpoint ...')
      if args.save_model :
        save_checkpoint(
                      {
                          'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          
                      },
                      checkpoint = args.checkpoint_path,
                      is_best=is_best)

    scheduler.step()
    logs['test_loss'].append(loss_test)
    logs['test_acc'].append(acc_test)

    print('Epoch [{}/{}],\ttrain loss:{:.4f},\ttrain acc:{:.2f},\ttest loss:{:.4f},\ttest accuracy:{:.2f},\t lr:{:.4f}'.format(epoch,args.num_epochs,train_loss,train_acc, loss_test, acc_test,optimizer_1.param_groups[0]['lr']))
  return logs



def cal_freq(train_loader):
    label_freq = {}
    for i, (_,target,_,_) in enumerate(train_loader):
        target = target.cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    return torch.tensor(label_freq_array)

def train_env(train_loader,loss_function,optimizer,model,classifer,scheduler,args):
  torch.manual_seed(seed)
  net = model.cuda()
  classifer = classifer.cuda()
  min_loss = 50 
  for epoch in tqdm(range(args.num_epochs)):
    loss = train_epoch_irm(net, classifer,loss_function, train_loader, optimizer,args)
    scheduler.step()
    is_best = True if (epoch>5) and loss < min_loss else False
    if is_best:
        min_loss = loss
        print(' => find better nosie indetifeir ...')
        save_checkpoint(
                  {
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      
                  },
                  checkpoint = args.classifer_path,
                  is_best=is_best)


def train_accuracy(outputs,labels):
    correct = 0
    _, pred = torch.max(outputs, 1)  
    correct += pred.eq(labels).cpu().sum().item() 
    return correct


def train_epoch_irm(net, classifer,loss_function, dataloader, optimizer):
    net.train()
    for i ,(sample,target,meta_reverse,meta_balance) in enumerate(dataloader):
        images_raw,images_reverse,images_balance = sample.cuda() , meta_reverse['sample_image'].cuda() , meta_balance['sample_image'].cuda()
        labels_raw,labels_reverse,labels_balance = target.cuda() , meta_reverse['sample_label'].cuda() , meta_balance['sample_label'].cuda()
        optimizer.zero_grad()
       
        feature_raw = net(images_raw)
        feature_reverse = net(images_reverse)
        feature_balance = net(images_balance)
        
        outputs_raw = classifer(feature_raw)
        outputs_reverse = classifer(feature_reverse)
        outputs_balance = classifer(feature_balance)

        penalty_raw = penalty(outputs_raw, labels_raw, loss_function)
        penalty_reverse = penalty(outputs_reverse, labels_reverse, loss_function)
        penalty_balance = penalty(outputs_balance, labels_balance, loss_function)
        env_penalty  = [penalty_raw,penalty_reverse,penalty_balance]
        irm_penalty = torch.stack(env_penalty).mean()
        loss = irm_penalty # add ce_loss

        loss.backward()
        optimizer.step()
    tro = classifer.module.get_tro()
    return loss
     #save the best



# get instance-weight

def get_weight_dict(dataloader,model,adjustment):
    model.cuda()
    model.eval()
    weight_dict ={}
    for i ,(images,labels,paths) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) - adjustment
        false_weight = 0.01
        # whether is true
        _, pred = torch.max(outputs, 1)
        softmax_layer = nn.Softmax(dim=1)
        outputs = softmax_layer(outputs)
        false_index =  [index for (index,value) in enumerate(pred.eq(labels)) if value == False]
        for index in false_index:
            weight_dict[paths[index]] = false_weight

        true_index = [index for (index,value) in enumerate(pred.eq(labels)) if value == True]
        for index in true_index:
            weight_dict[paths[index]] = float(torch.max(outputs[index]).cpu().detach().numpy())
        
    temp_value_list = []
    for key,value in weight_dict.items():
        if value != 0.01:
            temp_value_list.append(value)

    temp_value_list = np.array(temp_value_list)
    temp_value_list = temp_value_list.reshape(-1,1)
    tool = MinMaxScaler(feature_range=(0.5,1))
    normalize_list = tool.fit_transform(temp_value_list)

    #update
    i = 0
    for key,value in weight_dict.items():
        if value != 0.01:
            weight_dict[key] = float(normalize_list[i])
            i = i + 1
    
    return weight_dict

def get_confidence(dataloader,model,adjustment):
    model.cuda()
    for i ,(images,labels,paths) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) - adjustment
        _, pred = torch.max(outputs, 1)
        


if __name__ == "__main__":
    # Training settings


  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg',type = str ,default ='./noise_longtail/config/0.1_red.yaml')
  parser.add_argument('--num_epochs', type=int, default=180)
  parser.add_argument('--lr', type=float, default=0.2)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--device_ids', type=list, default=[0])
  parser.add_argument('--save_model',type = int,default=0)
  parser.add_argument('--randaug',type = int,default=0)
  parser.add_argument('--pic_name',type = str,default='normal.jpg')
  parser.add_argument('--model',type = str,default='resnet_18')
  args = parser.parse_args()

  # load config
  args = update_args(args)

  seed=123
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)


  #load path
  TRAIN_DATA_PATH = args.train_path
  CHECKPOINT_PATH = args.checkpoint_path
  JSON_PATH = args.json_path
  TEST_DATA_PATH = args.test_path
  RESAMPLE_PATH = args.resample_path

  pth_file_stage_1= args.pth_stage_1_path
  pth_file_stage_2= args.pth_stage_2_path
  FIANL_WEIGHT_PATH = args.final_weight_path
  CLASSIFER_PATH = args.classifer_path
  JSON_PATH = args.json_path


  if args.mix_up_weight:
    with open(JSON_PATH,'r') as load_f:
      score_dict = json.load(load_f)
      args.score_dict = score_dict
    

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

  #generate dataloader
  if args.resample ==1:
    sampler_dic =  {
                  'sampler': source_import(RESAMPLE_PATH).get_sampler(),
                  'params': {'num_samples_cls': 8}
              }

    my_sampler=sampler_dic['sampler'](train_data, **sampler_dic['params'])
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                              batch_size = args.batch_size,sampler = my_sampler,
                                              shuffle = False,num_workers=4)
  else:

    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                                batch_size = args.batch_size,
                                                shuffle = True,num_workers=4,pin_memory=True)


  test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                        batch_size = args.batch_size, 
                                        shuffle = False,num_workers=4)

  #load  raw  model
  if args.model =='resnet_50':
    model = models.resnet50(pretrained=False) 
  else:
    model = models.resnet18(pretrained=False) 
  model.fc= nn.Linear(in_features=model.fc.in_features, out_features=args.cls_num)
  resnet_model=nn.DataParallel(model,device_ids=[0,1,2,3])
  ce_loss = nn.CrossEntropyLoss().to(device)
  #ce_loss = NGCEandRCE(num_classes = args.cls_num,alpha = 0.5, beta = 0.5).to(device)


  logs_baseline = train_test_model(train_loader,test_loader,ce_loss,resnet_model,args)
  sub_train_loss= []
  sub_test_loss = []
  for item in logs_baseline['train_loss']:
      item = item.cpu().detach().numpy()
      sub_train_loss.append(item)

  logs_baseline['train_loss'] = sub_train_loss


  for item in logs_baseline['test_loss']:
      item = item.cpu().detach().numpy()
      sub_test_loss.append(item)

  logs_baseline['test_loss'] = sub_test_loss


  # save the training curve
  fig = plt.figure(figsize=(14,4))
  plt.subplot(1,2,1)
  plt.plot(logs_baseline['train_acc'], label='train', linewidth=2)
  plt.plot(logs_baseline['test_acc'], label='test', linewidth=2)
  plt.legend(frameon=False)
  plt.grid(True, color="#93a1a1", alpha=0.3)
  plt.ylabel("Accuracy", labelpad=15, fontsize=12, color="#333533", fontweight='bold');
  plt.xlabel("Epoch", labelpad=15, fontsize=12, color="#333533", fontweight='bold');

  plt.subplot(1,2,2)
  plt.plot(logs_baseline['train_loss'], label='Training loss', linewidth=2)
  plt.plot(logs_baseline['test_loss'], label='Validation loss', linewidth=2)
  plt.legend(frameon=False)
  plt.grid()
  plt.grid(True, color="#93a1a1", alpha=0.3)
  plt.xlabel("Epoch", labelpad=15, fontsize=12, color="#333533", fontweight='bold');
  plt.ylabel("Loss", labelpad=15, fontsize=12, color="#333533", fontweight='bold');
  plt.savefig(args.pic_name)


  print('learning_rate:' ,args.lr,'weight_decay',args.weight_decay,"\n")
  print('---------the best test acc :',max(logs_baseline['test_acc']),'-----------\n')





##########################noise identifier g(.)##############################################


## create muti-enviroment

irm_data = irmdataset(TRAIN_DATA_PATH, cls_num = args.cls_num)
irm_loader = torch.utils.data.DataLoader(dataset = train_data,
                                              batch_size = args.batch_size,
                                              shuffle = True,num_workers=4,pin_memory=True)

# load model
args.freq = cal_freq(train_loader)
model = models.resnet18(pretrained=False) 
model.fc= nn.Linear(in_features=model.fc.in_features, out_features=args.cls_num)
classifer = create_irm_classifer(args.freq,num_classes=args.cls_num,feat_dim=512)
resnet_model=nn.DataParallel(model,device_ids=[0,1,2,3])
resnet_classifer=nn.DataParallel(classifer,device_ids=[0,1,2,3])
model_dict= torch.load(pth_file_stage_1)
resnet_model.load_state_dict(model_dict['state_dict'])
resnet_model.module.fc = ReLU()
classifer_dict = resnet_classifer.state_dict()
checkpoint_dict = {k:v for k,v in model_dict['state_dict'].items() if k in classifer_dict}
classifer_dict.update(checkpoint_dict)
resnet_classifer.load_state_dict(classifer_dict)

#freeze 
for param_name, param in resnet_model.module.named_parameters():
    param.requires_grad = False
optimizer = optim.SGD([p for p in resnet_classifer.module.parameters() if p.requires_grad], lr=0.001, momentum=0.9,weight_decay = 1e-4)  
scheduler = StepLR(optimizer,step_size = 5 ,gamma = 0.9)
ce_loss = nn.CrossEntropyLoss().cuda()
# get class-balanced classifer
args.env_num = 3
train_env(irm_loader,ce_loss,optimizer,resnet_model,resnet_classifer,scheduler,args)



##########################get weight ##############################################

weight_data = imagenet_with_path(TRAIN_DATA_PATH,transform = None)
weight_loader = torch.utils.data.DataLoader(dataset = weight_data,
                                              batch_size = 512,
                                              shuffle = True,num_workers=0,pin_memory=True)


model = models.resnet18(pretrained=False) 
model.fc= nn.Linear(in_features=model.fc.in_features, out_features=args.cls_num)
model_dict= torch.load(pth_file_stage_2)
resnet_model=nn.DataParallel(model,device_ids=[0,1,2,3])
resnet_model.load_state_dict(model_dict['state_dict'])
adjustment = compute_adjustment(weight_loader, 1)
weight_dict = get_weight_dict(weight_loader,resnet_model,adjustment)
with open (FIANL_WEIGHT_PATH,'w') as f:
    json.dump(weight_dict,f)






