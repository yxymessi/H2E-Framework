import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.models as models
def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit

    
def create_model(m_type='resnet101'):
    # create various resnet models
    if m_type == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif m_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif m_type == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif m_type == 'resnext50':
        model = models.resnext50_32x4d(pretrained=False)
    elif m_type == 'resnext101':
        model = models.resnext101_32x8d(pretrained=False)
    else:
        raise ValueError('Wrong Model Type')
    model.fc = nn.ReLU()
    return model


class ClassifierLWS(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(ClassifierLWS, self).__init__()

        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

        self.scales = nn.Parameter(torch.ones(num_classes))
        for _, param in self.fc.named_parameters():
            param.requires_grad = False
        
    def forward(self, x, add_inputs=None):
        y = self.fc(x)
        y *= self.scales
        return y


def create_classifer(feat_dim=512, num_classes=100):
    model = ClassifierLWS(feat_dim=feat_dim, num_classes=num_classes)
    return model


class ClassifierIRM(nn.Module):
    def __init__(self, label_freq_array,feat_dim,num_classes):
        super(ClassifierIRM, self).__init__()

        self.fc = nn.Linear(feat_dim, num_classes, bias=True)
        self.tro = nn.Parameter(torch.ones(num_classes).cuda(), requires_grad= True)
        self.adjustments = torch.log(label_freq_array.pow(self.tro) + 1e-12)

    def forward(self, x, add_inputs=None):

        y = self.fc(x)-self.adjustments
        return y

    def get_tro(self):
        return self.tro


def create_irm_classifer(train_loader, num_classes,feat_dim = 512):
    model = ClassifierIRM(train_loader,feat_dim=feat_dim,num_classes=num_classes)
    return model



#LDAM classifer
class ClassifierLDAM(nn.Module):
    def __init__(self, feat_dim, num_classes=1000):
        super(ClassifierLDAM, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes).cuda(), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
        self.weight.data.renorm_(2, 1, 1e-5)
        self.weight.data.mul_(1e5)


    def forward(self, x, add_inputs=None):
        y = torch.mm(F.normalize(x, dim=1), F.normalize(self.weight, dim=0))
        return y


def create_model(feat_dim=2048, num_classes=100):
    model = ClassifierLDAM(feat_dim=feat_dim, num_classes=num_classes)
    return model