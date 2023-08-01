#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassPrecisionRecallCurve

import torchvision
import torchvision.utils
import torchvision.transforms as transforms

from PIL import Image
import time

import mmpretrain as mmp
from mmpretrain.models import backbones, classifiers, heads, necks

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

import matplotlib.pyplot as plt


class MicronNet(nn.Module):
    """
    MicronNet architecture based on the paper: https://arxiv.org/pdf/1804.00497.pdf
    Source: https://github.com/ppriyank/MicronNet/tree/master

    Parameters
    ----------
    linear_input_size : int
        Size of the input for the first fully connected layer.
    nclasses : int
        Number of output classes.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First 1x1 Convolution layer.
    conv0_bn : nn.BatchNorm2d
        Batch normalization for the input.
    conv1_bn : nn.BatchNorm2d
        Batch normalization for the first Convolution layer.
    conv2 : nn.Conv2d
        Second 5x5 Convolution layer.
    maxpool2 : nn.MaxPool2d
        MaxPool layer for the second Convolution layer.
    conv2_drop : nn.Dropout2d
        Dropout layer for the second Convolution layer.
    conv2_bn : nn.BatchNorm2d
        Batch normalization for the second Convolution layer.
    conv3 : nn.Conv2d
        Third 3x3 Convolution layer.
    maxpool3 : nn.MaxPool2d
        MaxPool layer for the third Convolution layer.
    conv3_drop : nn.Dropout2d
        Dropout layer for the third Convolution layer.
    conv3_bn : nn.BatchNorm2d
        Batch normalization for the third Convolution layer.
    conv4 : nn.Conv2d
        Fourth 3x3 Convolution layer.
    maxpool4 : nn.MaxPool2d
        MaxPool layer for the fourth Convolution layer.
    conv4_bn : nn.BatchNorm2d
        Batch normalization for the fourth Convolution layer.
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Second fully connected layer.
    dense1_bn : nn.BatchNorm1d
        Batch normalization for the first fully connected layer.

    Methods
    -------
    forward(x)
        Forward pass of the MicronNet model.

    """

    def __init__(self, linear_input_size, nclasses):
        super(MicronNet, self).__init__()
        
        # Layer 1: 1x1 Convolution
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv0_bn = nn.BatchNorm2d(3) # Batch normalization
        self.conv1_bn = nn.BatchNorm2d(1) 
        
        # Layer 2: 5x5 Convolution + MaxPool
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv2_drop = nn.Dropout2d()
        self.conv2_bn = nn.BatchNorm2d(29)
        
        # Layer 3: 3x3 Convolution + MaxPool
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv3_drop = nn.Dropout2d()
        self.conv3_bn = nn.BatchNorm2d(59)
        
        # Layer 4: 3x3 Convolution + MaxPool
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv4_bn = nn.BatchNorm2d(74)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(linear_input_size, linear_input_size//2)
        self.fc2 = nn.Linear(linear_input_size//2, nclasses)
        self.dense1_bn = nn.BatchNorm1d(linear_input_size//2) # Batch normalization for fully connected layer
        
    def forward(self, x):
        # Layer 1
        x =  F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        # Layer 2
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # Layer 3
        x = F.relu(self.conv3_bn(self.conv3( self.maxpool2(x))))
        # Layer 4
        x = F.relu(self.conv4_bn(self.conv4( self.maxpool3(x))))
        x = self.maxpool4(x)         
        # Flatten
        x = torch.flatten(x, start_dim=1) #x = x.view(x.size(0), -1)
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        out = F.dropout(x, training=self.training)
#         output = F.log_softmax(out)
        return out # raw outputs


class ResNet(nn.Module):
    """
    ResNet-based image classification model based on Open-MMLab
    Source: https://github.com/open-mmlab/mmpretrain
    
    The chossen network width is set to 32, you can modify it if necessary
    
    Types of ResNet-based models in OPen.MMLab framework
    # ResNet
    # Res2Net
    # SEResNet
    # ResNet_CIFAR
    # ResNeXt
    # SEResNeXt
    # ResNeSt

    Parameters
    ----------
    depth : int
        Depth of the ResNet model.
    head_in_channels : int
        Number of input channels to the head.
    head_mid_channels : int
        Number of intermediate channels in the head.
    num_classes : int
        Number of output classes.

    Attributes
    ----------
    resnet_backbone : backbones.ResNet_CIFAR
        ResNet backbone model.
    neck : necks.GlobalAveragePooling
        Global average pooling neck.
    head : heads.stacked_head.StackedLinearClsHead
        Stacked linear classification head for predictions.

    Methods
    -------
    forward(x)
        Forward pass of the ResNet model.

    """
    
    def __init__(self, depth, head_in_channels, head_mid_channels, num_classes):
        super(ResNet, self).__init__()
        
        STEM_CHANNELS=32  # Default width of the initial Convolution layer
        BASE_CHANNELS=32  # Default width of the first Convolution layer 
        self.resnet_backbone = backbones.ResNet_CIFAR(
            depth=depth, # model depth
            stem_channels=STEM_CHANNELS, # inital Conv layer width
            base_channels=BASE_CHANNELS, # changing model width
#             width_per_group=4 # only in case of ResNeXt
#             out_indices=(3,),
#             strides=(1, 2, 2, 2),
        )
        
        self.neck = necks.GlobalAveragePooling() # Average pooling Layer Neck
#         self.neck = DenseCLNeck(neck_in_channels, neck_mid_channels, neck_out_channels) # Dense Layer only Neck

        self.head = heads.stacked_head.StackedLinearClsHead(
            num_classes=num_classes,
            in_channels=head_in_channels,
            mid_channels=head_mid_channels,
            dropout_rate=0.5, # Dropout layer rate
        )

    def forward(self, x):
        x = self.resnet_backbone(x) # backbone
        x = self.neck(x) # neck
        x = self.head(x) # head
        output = F.log_softmax(x, dim=1) # apply logit to obtain distribution

        return output
    
   