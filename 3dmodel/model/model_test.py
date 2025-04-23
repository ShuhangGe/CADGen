import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils.parser import get_args
class UNet(nn.Module):
    def __init__(self, in_channel=96, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            #nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        #output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        #output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        #output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        #print('out1.shape: ',out.shape)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out2.shape: ',out.shape)
        output4 = self.map4(out)
        #print('output4.shape: ',output4.shape)
        return output4

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResnetBlock, self).__init__()
        self.conv0 = nn.Conv2d(3, in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition

        return y

class Views2Points(nn.Module):
    '''
    input : picture:(batchsize, channel, H, W)
    output: (batchsize, channel, H, W, Z)
    '''
    def __init__(self,config):
        super().__init__()
        self.img_feature =ResnetBlock(config.resnet_in,config.resnet_out,1)
        self.conv3d = nn.Conv3d(in_channels=96,
                        out_channels=96,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=0)
        self.unet = UNet(in_channel=config.resnet_out*3,out_channel=config.UNet_out)
        self.config = config


    def forward(self,side,front,top,cad_data):
        print('model start ')
        print('side.shape: ',side.shape)
        print('front.shape: ',front.shape)
        print('top.shape: ',top.shape)
        print('cad_data.shape: ',cad_data.shape)
        side_feature = self.img_feature(side)
        front_feature = self.img_feature(front)
        top_feature = self.img_feature(top)
        print('side_feature.shape: ',side_feature.shape)
        #print('front_features.shape: ',front_feature.shape)
        #print('top_feature.shape: ',top_feature.shape)
        assert side_feature.shape[-1]==front_feature.shape[-1]==top_feature.shape[-1]==side_feature.shape[-2]==front_feature.shape[-2]==top_feature.shape[-2]
        repeat_num = side_feature.shape[-1]
        side_3d = side_feature.unsqueeze(-3).repeat(1,1,repeat_num,1,1)
        front_3d = front_feature.unsqueeze(-2).repeat(1,1,1,repeat_num,1)
        top_3d = top_feature.unsqueeze(-1).repeat(1,1,1,1,repeat_num)
        print('side_3d.shape: ',side_3d.shape)
        #print('front_3d.shape: ',front_3d.shape)
        #print('top_3d.shape: ',top_3d.shape)
        feature_3d = torch.concat((side_3d,front_3d,top_3d),dim=1)
        print('feature_3d1.shape: ',feature_3d.shape)
        feature_3d = self.unet(feature_3d)
        print('feature_3d2.shape: ',feature_3d.shape)
        '''side_feature.shape:  torch.Size([1, 32, 64, 64])
        feature_3d1.shape:  torch.Size([1, 96, 64, 64, 64])
        feature_3d2.shape:  torch.Size([1, 32, 64, 64, 64])'''
        '''cad_data.shape:  torch.Size([50, 1024, 3])'''

        
        data = torch.cat([torch.cat([feature_3d[j,:,int((cad_data[j,i,0]+1)/2*64),int((cad_data[j,i,1]+1)/2*64),int((cad_data[j,i,2]+1)/2*64)].unsqueeze(0).unsqueeze(0)\
            for i in range(cad_data.shape[1])],dim=1) for j in range(feature_3d.shape[0])],dim=0)
        print('data.shape: ',data.shape)

        
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_args()
    side = torch.rand(50, 200, 200, 3).to(device)  
    front = torch.rand(50, 200, 200, 3).to(device)
    top = torch.rand(50, 200, 200, 3).to(device)
    cad_data = torch.rand(1, 1024, 3).to(device)
    '''
    side((x),y,z)
    front(x,(y),z)
    top:(x,y,(z))
    '''
    model = Views2Points(config=cfg).to(device)
    print(model)
    a= model(side,front,top,cad_data)
        
        