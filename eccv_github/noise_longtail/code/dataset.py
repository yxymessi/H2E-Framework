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
from randaugment import RandAugment, ImageNetPolicy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class imagenet_with_path(dset.ImageFolder):
 
    def __init__(self, root,transform):
        super(imagenet_with_path ,self).__init__(root, transform)
        self.indices = range(len(self)) 
        self.transform = transform


    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path) 
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,target,path


class ImageNetSolo(Dataset):
    def __init__(self, index, data_path):
        super(ImageNetSolo, self).__init__()
        categories = os.listdir(data_path)
        names = os.listdir(os.path.join(data_path, categories[index]))
        self.category = categories[index]
        self.images = [os.path.join(data_path, categories[index], name) for name in names]
        self.transform = transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.images[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            sample = self.transform(sample)
        return sample, path




#irmdataset

class irmdataset(dset.ImageFolder):
    def __init__(self, root, cls_num):
        super(irmdataset ,self).__init__(root,cls_num)
        self.indices = range(len(self)) 
        self.cls_num = cls_num
        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        self.class_dict = self._get_class_dict()
        self.transform = None
        self.transform_reverse  = transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.transform_uniform  =    transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        # raw, balance, reverse
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        dic=self.class_to_idx
        meta_balance = dict()
        meta_reverse = dict()

        sample_class = self.sample_class_index_by_weight()
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_path,sample_label = self.samples[sample_index]
        sample_img = self.transform_reverse(self.loader(sample_path))
        meta_reverse['sample_image'] = sample_img
        meta_reverse['sample_label'] = sample_label
        
        
        sample_class = random.randint(0, self.cls_num-1)
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_path,sample_label = self.samples[sample_index]
        sample_img = self.transform_uniform(self.loader(sample_path))
        meta_balance['sample_image'] = sample_img
        meta_balance['sample_label'] = sample_label

        return sample,target,meta_reverse,meta_balance
    

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos
    
    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

