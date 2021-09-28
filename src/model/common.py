import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from model.adder import adder2d
# from model.adder import Power_Activation

def default_conv(in_channels, out_channels, kernel_size, padding = 1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Power_Activation(nn.Module):
    def __init__(self):
        super(Power_Activation,self).__init__()
        self.a = nn.Parameter(torch.ones(1))

    def forward(self,x):
        return torch.sin(x)*torch.pow(torch.abs(x),self.a)


def adder(in_channels, out_channels, kernel_size = 3, stride=1, bias=False ):
  " 3x3 convolution with padding "
  return adder2d(in_channels, out_channels, kernel_size= kernel_size, stride=stride, padding=1, bias=bias)

def pointwise(in_channels, out_channels, kernel_size = 1, stride=1, bias=False ):
  " 1x1 convolution with padding "
  return adder2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)

def depthwise(in_channels, out_channels, kernel_size, bias=False):
   return adder2d(in_channels, out_channels, kernel_size, padding=1, group = 1,bias=bias)

# def Power_activation(in_channels, out_channels, kernel_size):
    # return Power_Activation(in_channels, out_channels, kernel_size)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = depthwise(nin, nin, kernel_size=kernel_size, bias=bias)
        #self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        #self.pointwise = nn.Conv2d(nin, nout, kernel_size = 1, stride=1, bias=False )
        self.pointwise = pointwise(nin, nout, kernel_size = 1, stride=1, bias=bias )

    def forward(self, x):
        out = self.depthwise(x)
        #print("x:", x.shape)
        out = self.pointwise(out)
        return out

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, stride, bias=bias)] #bias=bias # stride
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, stride,  # stride
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1): #bias=True
        super(ResBlock, self).__init__()
        m = []
        # for i in range(2):
            # if i == 0:
            #     m.append(conv(n_feats, n_feats, kernel_size, stride, bias = bias)) #bias=bias # stride
            # if i == 1:
            #     m.append(pointwise(n_feats, n_feats, kernel_size = 1, stride=1, bias=bias ))
            # if i == 2:
            #     m.append(pointwise(n_feats, n_feats, kernel_size = 1, stride=1, bias=bias ))                
            # if bn:
            #     m.append(nn.BatchNorm2d(n_feats))
            # if i == 0:
            #     m.append(act)
       
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        #n = []
        #n.append(Power_activation(n_feats, n_feats, kernel_size))
        #self.abc = nn.Sequential(*n)
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        # print("x", x.shape)
        # print("res", res.shape)
        #res = self.abc(res)
        
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=True, act=False, bias=True): #bias=True

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, bias)) #bias  # stride
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, 1, bias)) #bias  # stride 1
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

