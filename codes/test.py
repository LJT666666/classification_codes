#coding:utf-8
import numpy as np
import os
import cv2
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import copy
import json
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(MyDataset, self).__init__()

        dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                'ship': 8, 'truck': 9}
        imglb = []
        path = []

        file_info = pd.read_csv('all/trainLabels.csv')
        # print(file_info)
        # print(file_info.loc[:, 'label'])
        file_class = file_info['label']

        for each in range(len(file_class)):
            a = dict.get(file_class[each])
            imglb.append(a)

        imglb = imglb[0:512]


        for i in range(1500):
            imgpath = ('all/train' + '/' + str(i) + '.png')
            path.append(imgpath)
            #print(path)
        path = path[0:1500]

        self.image = path
        self.imglb = imglb
        self.root = root
        self.size = 1500
        self.transform = transform

    def __getitem__(self, index):
        print(self.image[index])
        img = Image.open(self.image[index])

        label = self.imglb[index]
        sample = {'image': img, 'classes': label}
        if self.transform:
            image =self.transform(img)
            sample['image'] = image

        return sample


    def __len__(self):
        return self.size

train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                       ])
val_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ])

TRAIN_ANNO = 'Classes_train_annotation.csv'
VAL_ANNO = 'Classes_val_annotation.csv'
CLASSES = ['0', '1','2','3','4','5', '6','7','8','9']
train_dataset = MyDataset(root = TRAIN_ANNO,transform = train_transforms)
test_dataset = MyDataset(root = VAL_ANNO,transform = val_transforms)
#设置batch size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=512,shuffle=True,drop_last = True )
data_loaders = {'train': train_loader, 'val': test_loader}

classes = ['0', '1','2','3','4','5', '6','7','8','9']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

print("50bestbbb4")
# net = ResNet18()
# net = torch.load('ckpt.pkl')
net = torch.load('cifar10(50)bb4bestbest.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
correct = 0
total = 0
with torch.no_grad():
    # for data in test_loader:
    #     images, labels = data['image'], data['classes']
    #     # print(labels)
    #     images, labels = images.to(device), labels.to(device)
        loader = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
        'ship': 8, 'truck': 9}
        imglb = []
        imglb2 = []
        file_info = pd.read_csv('all/trainLabels.csv')
        # print(file_info)
        # print(file_info.loc[:, 'label'])
        file_class = file_info['label']

        # print(file_class[0:5000])
        for each in range(len(file_class)):
            a = dict.get(file_class[each])
            imglb.append(a)
        # print(imglb[0:5000])

        imglb2 = torch.Tensor(imglb)

        # txts = [[] for _ in range(5084)]
        # txts = [[i for i in range(5186)]]
        txts = []
        k = 0

        predictedall = []
        predictedall = torch.Tensor(predictedall)

        batch_size =512
        n = 5000 // batch_size
        for j in range(n+1):
            imglb = imglb2
            print('第 %d 个512' % (j+1))
            if j != n:
                imglb = imglb[j*512:(j+1)*512]

            # if j == 0:
            #     imglb = imglb[0:512]
            # if j == 1:
            #     imglb = imglb[512:1024]
            #     print(imglb)
            BS=512
            if j==n:
                BS = 5000 % batch_size
                print(BS)
                imglb = imglb[4608:j*512+BS]

            img = Image.open("all/train/"+str(j*512)+".png")
            image = loader(img).unsqueeze(0)
            for i in range(j*512+1,j*512+BS):
                img = Image.open("all/train/"+str(i)+".png")
                image = torch.cat((image, loader(img).unsqueeze(0)), dim=0)

            image.to(device, torch.float)
            image = image.to(device)
            outputs = net(image)

            # print(outputs.item())
            _, predicted = torch.max(outputs.data, 1)
            predicted=predicted.to("cpu")
            predictedall = torch.cat((predictedall, predicted), dim=0)
            # print(imglb)
            print(predicted)
            numbers = []
            if j!=n:
                for h in range(512):
                    # numbers.append(predicted[h].item())
                    txts.append(predicted[h].item())
                    # txts += (list(predicted.numpy()))

                    filename = '../classesnames/' + str(k) + '.txt'  # 批量生成的文件名

                    f = open(filename, 'w')
                    f.write(str(txts[k]))
                    f.close()
                    k = k + 1
                    # print(predicted[h].item())
            else:
                for h in range(392):
                    numbers.append(predicted[h].item())
                    txts.append(predicted[h].item())
                    # txts += (list(predicted.numpy()))

                    filename = '../classesnames/' + str(k) + '.txt'  # 批量生成的文件名

                    f = open(filename, 'w')
                    f.write(str(txts[k]))
                    f.close()
                    k = k + 1

            total += imglb.size(0)

            correct += (predicted == imglb).sum().item()

            print('Accuracy of the network on the 5000 test images: %d %%' % (
            100 * correct /total))

