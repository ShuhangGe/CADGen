import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import cv2
from PIL import Image
import torch.nn.functional as F

class BasicBlock_18(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock_18, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck_50(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(Bottleneck_50,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=stride[2],padding=padding[2],bias=False),
            nn.BatchNorm2d(out_channels*4)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        # if stride[1] != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         # 卷积核为1 进行升降维
        #         # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
        #         nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet50(nn.Module):
    def __init__(self, resnet_out=256, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        Bottleneck = Bottleneck_50
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck,64,[[1,1,1]]*3,[[0,1,0]]*3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck,128,[[1,2,1]] + [[1,1,1]]*3,[[0,1,0]]*4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck,256,[[1,2,1]] + [[1,1,1]]*5,[[0,1,0]]*6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck,resnet_out,[[1,2,1]] + [[1,1,1]]*2,[[0,1,0]]*3)
        #self.conv5 = self._make_layer(Bottleneck,resnet_out,[[1,1,1]] + [[1,1,1]]*2,[[0,1,0]]*3)



    def _make_layer(self,block,out_channels,strides,paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0,len(strides)):
            layers.append(block(self.in_channels,out_channels,strides[i],paddings[i],first=flag))
            flag = False
            self.in_channels = out_channels * 4
            

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        print('out1.shape: ',out.shape)
        out = self.conv2(out)
        print('out2.shape: ',out.shape)
        out = self.conv3(out)
        print('out3.shape: ',out.shape)
        out = self.conv4(out)
        print('out4.shape: ',out.shape)
        out = self.conv5(out)
        print('out5.shape: ',out.shape)
        ''''''
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, resnet_out=256, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        BasicBlock = BasicBlock_18
        self.in_channels = 64        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,resnet_out,[[2,1],[1,1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        print('out1.shape: ',out.shape)
        out = self.conv2(out)
        print('out2.shape: ',out.shape)
        out = self.conv3(out)
        print('out3.shape: ',out.shape)
        out = self.conv4(out)
        print('out4.shape: ',out.shape)
        out = self.conv5(out)
        print('out5.shape: ',out.shape)
        #out = self.avgpool(out)
        
        print('out6.shape: ',out.shape)

        return out


if __name__ == '__main__':
    # 定义网络
    res50 = ResNet50(resnet_out=256)
    pic = torch.rand((1, 3, 256, 256))
    out = res50(pic)
    print(out.shape)
    #print(res50)
    
