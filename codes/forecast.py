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


print("70")
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

        # txts = [[] for _ in range(5084)]
        # txts = [[i for i in range(5186)]]
        txts = []
        k = 0

        predictedall = []
        predictedall = torch.Tensor(predictedall)

        batch_size =512
        n = 5000 // batch_size
        for j in range(n+1):

            print('第 %d 个512' % (j+1))

            BS=512
            if j==n:
                BS = 5000 % batch_size
                print(BS)


            img = Image.open("all/test/"+str(j*512)+".png")
            image = loader(img).unsqueeze(0)
            for i in range(j*512+1,j*512+BS):
                img = Image.open("all/test/"+str(i)+".png")
                image = torch.cat((image, loader(img).unsqueeze(0)), dim=0)

            image.to(device, torch.float)
            image = image.to(device)
            outputs = net(image)

            # print(outputs.item())
            _, predicted = torch.max(outputs.data, 1)
            predicted=predicted.to("cpu")
            predictedall = torch.cat((predictedall, predicted), dim=0)

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
