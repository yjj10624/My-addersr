'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

def adder2d_function(X, W, stride=1, padding=0, group = 0):
    n_filters, d_filter, h_filter, w_filter = W.size() 
    n_x, d_x, h_x, w_x = X.size()  

    # print("X.shape",X.shape)
    # print("aa:", type(torch.chunk(X, n_x, dim=0)))
    # print("W.shape",W.shape)    
    # if group == 1:
    #     for x, w in zip(torch.chunk(X, d_x, dim=1),torch.chunk(W, d_filter, dim=1)):
    #         n_x, d_x, h_x, w_x = x.size()
    #         n_filters, d_filter, h_filter, w_filter = w.size()  
    #         print("x:",x.shape)
    #         print("w:",w.shape)
    #         h_out = (h_x - h_filter + 2 * padding) / stride + 1
    #         w_out = (w_x - w_filter + 2 * padding) / stride + 1
    #         h_out, w_out = int(h_out), int(w_out)
            
    #         X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    #         # print("ss", torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).shape)
    #         #print("ss1", X_col.shape)
    #         X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)

    #         W_col = w.view(n_filters, -1)
    #         #print("X.shape",X_col.shape)
    #         #print("W.shape",W_col)
    #         out = adder.apply(W_col,X_col)
    #         #print("out:", out)
            
    #         out = out.view(n_filters, h_out, w_out, n_x)
    #         # print("out1 : ", out.shape)
    #         out = out.permute(3, 0, 1, 2).contiguous()
    #         out_list = []
    #         out_list.append(out)
    #         #print("out2 : ", out.shape)

    #     output = torch.cat(out_list,dim=1)
    #     # print("out2 : ", output.shape)
    #     return output


    # x = []
    # w = []
    
    #     for i in range(n_x):
    #         x[i] = torch.chunk(X, n_x, dim=0)[2]
    #     for j in range(n_filters):
    #         w[j] = torch.chunk(W, n_filters, dim=0)[j]
    # X1, X2, X3, X4 = X.chunk(n_x, dim=0)
    # W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15, W16, \
    #     W17, W18, W19, W20, W21, W22, W23, W24, W25, W26, W27, W28, W29, W30, W31, W32, \
    #         W33, W34, W35, W36, W37, W38, W39, W40, W41, W42, W43, W44, W45, W46, W47, W48, \
    #             W49, W50, W51, W52, W53, W54, W55, W56, W57, W58, W59, W60, W61, W62, W63, W64 \
    #                 = W.chunk(n_filters, dim=0)
    # X_list = [X1, X2, X3, X4]
    # W_list = [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15, W16, \
    #     W17, W18, W19, W20, W21, W22, W23, W24, W25, W26, W27, W28, W29, W30, W31, W32, \
    #         W33, W34, W35, W36, W37, W38, W39, W40, W41, W42, W43, W44, W45, W46, W47, W48, \
    #             W49, W50, W51, W52, W53, W54, W55, W56, W57, W58, W59, W60, W61, W62, W63, W64 ]

    if group == 1:

        out_list = []
        for x,w in zip(X.chunk(n_x, dim=0),W.chunk(n_x, dim=0)):
            n_x1, d_x, h_x, w_x = x.size()
            n_filters1, d_filter, h_filter, w_filter = w.size()  

            h_out = (h_x - h_filter + 2 * padding) / stride + 1
            w_out = (w_x - w_filter + 2 * padding) / stride + 1
            h_out, w_out = int(h_out), int(w_out)

            X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x1, -1, h_out*w_out)
            #print("X.shape",X_col.shape)
            X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)

            W_col = w.view(n_filters1, -1)
            # print("X.shape",X_col.shape)
            #print("W.shape",W_col.shape)
            out = adder.apply(W_col,X_col)
            
            out = out.view(n_filters1, h_out, w_out, n_x1)

            out = out.permute(3, 0, 1, 2).contiguous()

            out_list.append(out)
            # w_list.append(w)
            # x_list.append(x)

        for i in range(1, len(out_list)):
            out_list[0] = torch.cat((out_list[0],out_list[i]),dim=1)
        output = out_list[0]
        output = torch.cat((output,output), dim = 0)
        output = torch.cat((output,output), dim = 0)
        #print("out_put:",output.shape)
        # for j in range(1, len(w_list)):
        #     w_list[0] = torch.cat((w_list[0],w_list[j]),dim=1)
        # w = w_list[0]
        # #print("w:",w.shape)

        # for k in range(1, len(x_list)):
        #     x_list[0] = torch.cat((x_list[0],x_list[k]),dim=1)
        # x = x_list[0]
        # #print("x:",x.shape)
        return output

    if group == 0 :
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1
        h_out, w_out = int(h_out), int(w_out)
        
        X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        # print("ss", torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).shape)
        # print("ss1", X_col.shape)
        #print("X.shape",X_col.shape)
        X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)

        W_col = W.view(n_filters, -1)
        # print("X.shape",X_col.shape)
        # print("W.shape",W_col.shape)
        out = adder.apply(W_col,X_col)
        #print("out:", out)
        
        out = out.view(n_filters, h_out, w_out, n_x)
        # print("out1 : ", out.shape)
        out = out.permute(3, 0, 1, 2).contiguous()
        #print("out2 : ", out.shape)
        return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, group = 0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.group = group
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(int(output_channel),int(input_channel),kernel_size,kernel_size)))  #divided by group?
        self.bias = bias

        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.stride, self.padding, self.group)

        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output

###################################################################################################################################
# HIGH-PASS FILTER 만들기

def power_activation(X, W, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.size() 
    n_x, d_x, h_x, w_x = X.size()    
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)

    W_col = W.view(n_filters, -1)

    # print("X.shape",X_col.shape)
    # print("W.shape",W_col.shape)
    output = activation.apply(X_col, W_col)

    # print("OUTPUT.shape",output.shape)
    output = output.view(n_filters, 24*h_out, 24*w_out, n_x)
    # print("out : ", out.shape)
    output = output.permute(3, 0, 1, 2).contiguous()
    # print("out : ", output.shape)

    return output

class activation(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)

        output = torch.sin(X_col.unsqueeze(2))*torch.pow(torch.abs(X_col.unsqueeze(2)),W_col.unsqueeze(0))

        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(2)-W_col.unsqueeze(0))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(2)-W_col.unsqueeze(0)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)

        return grad_W_col, grad_X_col

class Power_Activation(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size):
        super(Power_Activation,self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size        
        self.W = torch.nn.Parameter(nn.init.normal_(torch.randn(int(output_channel),int(input_channel),kernel_size,kernel_size)))
        #self.W.requires_grad = True

    def forward(self, X):

        output = power_activation(X, self.W)

        return output


