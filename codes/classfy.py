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
from efficientnet_pytorch import EfficientNet


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
        imglb = imglb[5000:45000]

        for i in range(45000):
            imgpath = ('all/train' + '/' + str(i) + '.png')
            path.append(imgpath)
        path = path[5000:45000]

        self.image = path
        self.imglb = imglb
        self.root = root
        self.size = 40000
        self.transform = transform

    def __getitem__(self, index):
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
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                       ])
val_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ])

TRAIN_ANNO = 'Classes_train_annotation.csv'
VAL_ANNO = 'Classes_val_annotation.csv'
CLASSES = ['0', '1','2','3','4','5', '6','7','8','9']
train_dataset = MyDataset(root = TRAIN_ANNO,transform = train_transforms)
test_dataset = MyDataset(root = VAL_ANNO,transform = val_transforms)
#设置batch size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=256, num_workers=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, num_workers=10)
data_loaders = {'train': train_loader, 'val': test_loader}

classes = ['0', '1','2','3','4','5', '6','7','8','9']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 245   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 256      #批处理尺寸(batch_size)
LR = 0.01      #学习率

def ResNet18():
    # model = EfficientNet.from_name('efficientnet-b4')
    # in_fea = model._fc.in_features
    # model._fc = nn.Linear(in_features=in_fea, out_features=10, bias=True)
    model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=10)

    res18 = models.resnet18(pretrained=True)
    res18.fc = nn.Linear(512,10)
    # return res50
    return model
    # return ResNet(ResidualBlock)

net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
scheduler = optim.lr_scheduler.StepLR(optimizer,35,gamma=0.1,last_epoch=-1)

# 训练
if __name__ == "__main__":


    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data['image'], data['classes']
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    print(optimizer.param_groups[0]['lr'])
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                if(epoch > 140):
                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in test_loader:
                            net.eval()
                            inputs, labels = data['image'], data['classes']
                            images, labels = inputs.to(device), labels.to(device)
                            outputs = net(images)
                            # 取得分最高的那个类 (outputs.data的索引号)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                        acc = 100. * correct / total
            # state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            # torch.save(state,'save.pt')
                if (epoch % 5) ==0:
                    print(epoch % 5)
                    torch.save(net, 'cifar10('+ str(epoch) +')bbb4.pkl')
                    net = torch.load('cifar10('+  str(epoch) +')bbb4.pkl')
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

                        dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                                'horse': 7,
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

                        batch_size = 512
                        n = 5000 // batch_size
                        for j in range(n + 1):
                            imglb = imglb2
                            print('第 %d 个512' % (j + 1))
                            if j != n:
                                imglb = imglb[j * 512:(j + 1) * 512]

                            # if j == 0:
                            #     imglb = imglb[0:512]
                            # if j == 1:
                            #     imglb = imglb[512:1024]
                            #     print(imglb)
                            BS = 512
                            if j == n:
                                BS = 5000 % batch_size
                                print(BS)
                                imglb = imglb[4608:j * 512 + BS]

                            img = Image.open("all/train/" + str(j * 512) + ".png")
                            image = loader(img).unsqueeze(0)
                            for i in range(j * 512 + 1, j * 512 + BS):
                                img = Image.open("all/train/" + str(i) + ".png")
                                image = torch.cat((image, loader(img).unsqueeze(0)), dim=0)

                            image.to(device, torch.float)
                            image = image.to(device)
                            outputs = net(image)

                            # print(outputs.item())
                            _, predicted = torch.max(outputs.data, 1)
                            predicted = predicted.to("cpu")
                            predictedall = torch.cat((predictedall, predicted), dim=0)
                            # print(imglb)
                            # print(predicted)
                            numbers = []
                            # if j!=n:
                            #     for h in range(512):
                            #         # numbers.append(predicted[h].item())
                            #         txts.append(predicted[h].item())
                            #         # txts += (list(predicted.numpy()))
                            #
                            #         filename = '../classesnames/' + str(k) + '.txt'  # 批量生成的文件名
                            #
                            #         f = open(filename, 'w')
                            #         f.write(str(txts[k]))
                            #         f.close()
                            #         k = k + 1
                            #         # print(predicted[h].item())
                            # else:
                            #     for h in range(392):
                            #         numbers.append(predicted[h].item())
                            #         txts.append(predicted[h].item())
                            #         # txts += (list(predicted.numpy()))
                            #
                            #         filename = '../classesnames/' + str(k) + '.txt'  # 批量生成的文件名
                            #
                            #         f = open(filename, 'w')
                            #         f.write(str(txts[k]))
                            #         f.close()
                            #         k = k + 1

                            total += imglb.size(0)

                            correct += (predicted == imglb).sum().item()

                            print('Accuracy of the network on the 5000 test images: %d %%' % (
                                    100 * correct / total))

                #torch.save(net, 'cifar10(10)bbb4.pkl')
                # if epoch == 20:
                #     torch.save(net, 'cifar10(20)bbb4.pkl')
                # if epoch == 25:
                #     torch.save(net, 'cifar10(25)bbb4.pkl')
                # if epoch == 30:
                #     torch.save(net, 'cifar10(30)bbb4.pkl')
                # if epoch == 40:
                #     torch.save(net, 'cifar10(40)bbb4.pkl')
                # if epoch == 50:
                #     torch.save(net, 'cifar10(50)bbb4.pkl')
                # if epoch == 60:
                #     torch.save(net, 'cifar10(60)bbb4.pkl')
                # if epoch == 70:
                #     torch.save(net, 'cifar10(70)bbb4.pkl')
                # if epoch == 80:
                #     torch.save(net, 'cifar10(80)bbb4.pkl')
                # if epoch == 90:
                #     torch.save(net, 'cifar10(90)bbb4.pkl')
                # if epoch == 100:
                #     torch.save(net, 'cifar10(100)bbb4.pkl')
                # if epoch == 110:
                #     torch.save(net, 'cifar10(110)bbb4.pkl')
                # if epoch == 120:
                #     torch.save(net, 'cifar10(120)bbb4.pkl')
                # if epoch == 130:
                #     torch.save(net, 'cifar10(130)bbb4.pkl')
                # if epoch == 140:
                #      torch.save(net, 'cifar10(140)bbb4.pkl')
                # if epoch == 150:
                #     torch.save(net, 'cifar10(150)bbb4.pkl')
                # if epoch == 160:
                #     torch.save(net, 'cifar10(160)bbb4.pkl')
                # if epoch == 170:
                #     torch.save(net, 'cifar10(170)bbb4.pkl')
                # if epoch == 180:
                #     torch.save(net, 'cifar10(180)bbb4.pkl')
                # if epoch == 190:
                #     torch.save(net, 'cifar10(190)bbb4.pkl')
                # if epoch == 200:
                #     torch.save(net, 'cifar10(200)bbb4.pkl')
                # if epoch == 210:
                #     torch.save(net, 'cifar10(210)bbb4.pkl')
                # if epoch == 220:
                #     torch.save(net, 'cifar10(220)bbb4.pkl')
                # if epoch == 230:
                #     torch.save(net, 'cifar10(230)bbb4.pkl')
                # if epoch == 240:
                #     torch.save(net, 'cifar10(240)bbb4.pkl')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

