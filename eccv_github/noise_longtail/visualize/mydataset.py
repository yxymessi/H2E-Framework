import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torchvision.datasets as dset

class Mydataset(dset.ImageFolder):
    def __init__(self, root, transform=None):
        super(Mydataset ,self).__init__(root,transform)
        self.indices = range(len(self)) 
        self.cls_num = 100
        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        self.class_dict = self._get_class_dict()
        
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
        sample_img = self.transform(self.loader(sample_path))
        meta_reverse['sample_image'] = sample_img
        meta_reverse['sample_label'] = sample_label
        
        
        sample_class = random.randint(0, self.cls_num-1)
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_path,sample_label = self.samples[sample_index]
        sample_img = self.transform(self.loader(sample_path))
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
    