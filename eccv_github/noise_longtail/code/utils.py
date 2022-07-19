import os
import sys
import time
import math
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.utils.data as data
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import yaml
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#mix up

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



#mix up with weight


def mixup_data_weight(x, y, score_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    score_list,new_score_list = generate_pair_list(score_list,index)
    mixed_x = score_list.view(batch_size,1,1,1).to(device) * x + new_score_list.view(batch_size,1,1,1).to(device) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, score_list,new_score_list

def generate_pair_list(score_list,index):
    new_score_list =[]
    for k in index:
        new_score_list.append(score_list[k])
    sum_list = list(map(lambda x :x[0]+x[1] ,zip(score_list,new_score_list)))
    for j in range(len(sum_list)):
        score_list[j] = score_list[j]/sum_list[j]
        new_score_list[j] = new_score_list[j]/sum_list[j]
    score_list = torch.tensor(score_list)
    new_score_list = torch.tensor(new_score_list)
    return score_list,new_score_list




def get_weight_index(feature,path):
    feature = feature.cpu().numpy()
    sim_matrix= cosine_similarity(feature)
    rowSum = np.sum(sim_matrix, axis=1)
    index_list = rowSum.argsort()
    path_list = []
    for index in index_list:
        path_list.append(path[index])
    return path_list



def save_checkpoint(state,
                    checkpoint,
                    filename='checkpoint.pth.tar',
                    is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath,
                        os.path.join(checkpoint, 'checkpoint_best.pth.tar'))

def update_args(args):

    with open (args.cfg,encoding = 'utf-8') as f:
        config = yaml.load(f.read(),Loader = yaml.FullLoader)

    args.train_path = config['TRAIN_DATA_PATH']
    args.test_path = config['TEST_DATA_PATH']
    args.resample_path = config['RESAMPLE_PATH']
    args.checkpoint_path = config['CHECKPOINT_PATH']
    args.classifer_path = config['CLASSIFER_PATH']
    args.pth_stage_1_path = config['PTH_STAGE_1_PATH']
    args.pth_stage_2_path = config['PTH_STAGE_2_PATH']
    args.json_path =  config['JSON_PATH']
    args.final_weight_path = config['FIANL_WEIGHT_PATH']
    args.final_model_path  = config['FINAL_MODEL_PATH']


    #iterative
    args.iter_checkpoint_path = config['ITER_CHECKPOINT_PATH']
    args.iter_classifer_path = config['ITER_CLASSIFER_PATH']
    args.iter_pth_stage_1_path = config['ITER_PTH_STAGE_1_PATH']
    args.iter_pth_stage_2_path = config['ITER_PTH_STAGE_2_PATH']
    args.iter_json_path =  config['ITER_JSON_PATH']
    args.iter_final_weight_path = config['ITER_FIANL_WEIGHT_PATH']
    args.iter_final_model_path  = config['ITER_FINAL_MODEL_PATH']


    args.batch_size = config['batch_size']
    args.cls_num = config['cls_num']
    args.logit_adjust = config['logit_adjust']
    args.resample = config['resample']
    args.mix_up = config['mix_up']
    args.mix_up_weight = config['mix_up_weight']
    return args



# cleanlab
def get_probs(loader, model):
    # Switch to evaluate mode.
    model.cuda()
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            #print("\rComplete: {:.1%}".format(i / n_total), end="")

            input = input.cuda()

            # compute output
            outputs.append(model(input))

    # Prepare outputs as a single matrix
    probs = np.concatenate([
        torch.nn.functional.softmax(z, dim=1).cpu().numpy()
        for z in outputs
    ])

    return probs


#create_optimizer

def create_optimizer(model, classifier, lr,weight_decay,momentum):
    all_params = []

    for _, val in model.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [{"params": [val], "lr": lr, "weight_decay": weight_decay,"momentum" :momentum}]
    for _, val in classifier.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [{"params": [val], "lr": lr, "weight_decay": weight_decay,"momentum" :momentum}]
    return optim.SGD(all_params)



def count_dataset(train_loader):
    label_freq = {}
    for label in train_loader.dataset.targets:
        key = str(label)
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    return label_freq_array




def calculate_reverse_instance_weight(dataloader):
    # counting frequency
    label_freq = {}
    for key in dataloader.dataset.targets:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = torch.FloatTensor(list(label_freq.values()))
    reverse_class_weight = label_freq_array.max() / label_freq_array
    # generate reverse weight
    reverse_instance_weight = torch.zeros(len(dataloader.dataset)).fill_(1.0)
    for i, label in enumerate(dataloader.dataset.targets):
        reverse_instance_weight[i] = reverse_class_weight[label] / (label_freq_array[label] + 1e-9)
    return reverse_instance_weight




class DistributionSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float() # init weight


    def __iter__(self):
        self.prob = self.weight / self.weight.sum()

        indices = torch.multinomial(self.prob, self.num_samples, replacement=True).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()


#irm v1
def penalty(logits, y, loss_function):
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


#irm v2 rex

def info_nce_loss(features, batch_size, temperature):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    # features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels



