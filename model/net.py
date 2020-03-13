import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_extract import FeatureExtractNet


class FCN(nn.Module):
    """
    This network using to making upsampling and predicting for Fully Convolutional Neural Network
    """
    def __init__(self):
        """
        define a de-convolutional process for feature map
        """
        super(FCN, self).__init__()
        # self.num_class = num_class  #TODO: config with config file
        self.features = FeatureExtractNet()
        
        # transpose_conv8
        self.transpose_conv8 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 0)
        
        # transpose_conv9
        self.transpose_conv9 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)

        # transpose_conv10
        self.transpose_conv10 = nn.ConvTranspose2d(64, 16, kernel_size = 2, stride = 2, padding = 0)

        # transpose_conv11
        self.transpose_conv11 = nn.ConvTranspose2d(16, 16, kernel_size = 4, stride = 4, padding = 0)

        # refinement_conv12
        self.conv12_1 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_1 = nn.BatchNorm2d(32)
        self.conv12_2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_2 = nn.BatchNorm2d(32)
        self.conv12_3 = nn.Conv2d(32, 8, kernel_size = 1, stride = 1, padding = 0)
        self.dropout12 = nn.Dropout(p=0.3)
        self.conv12_4 = nn.Conv2d(8, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn12_4 = nn.BatchNorm2d(32)
        self.conv12_5 = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn12_5 = nn.BatchNorm2d(16)

        # classification13
        self.conv13 = nn.Conv2d(16, 1, kernel_size = 1, stride = 1, padding = 0)
        self.upscale_13 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            x(Variable): contains a batch of images with dimension batch_size x channels x H x W

        Returns:
            out(Variable): value of each pixel(probability) 
            
        Note the dimensions after each step are provided
        """
        #input                                                           batch_size x 2 x 256 x 256
        # feature map extraction
        features = self.features(x)

        # transpose_conv8 
        x = F.relu(self.transpose_conv8(features[-1]) + features[4])    # batch_size x 128 x 8 x 8
       
        # transpose_conv9
        x = F.relu(self.transpose_conv9(x) + features[3])               # batch_size x 64 x 16 x 16
        
        # transpose_conv10
        x = F.relu(self.transpose_conv10(x) + features[2])              # batch_size x 16 x 32 x 32
        
        # transpose_conv11
        x = F.relu(self.transpose_conv11(x))                            # batch_size x 16 x 128 x 128
    
        # refinement12
        x = F.relu(self.bn12_1(self.conv12_1(x)))                       # batch_size x 32 x 128 x 128
        x = F.relu(self.bn12_2(self.conv12_2(x)))                       # batch_size x 32 x 128 x 128
        x = torch.sigmoid(self.conv12_3(x))                                 # batch_size x 8 x 128 x 128
        x = self.dropout12(x)                                           # batch_size x 8 x 128 x 128
        x = F.relu(self.bn12_4(self.conv12_4(x)))                       # batch_size x 32 x 128 x 128
        x = F.relu(self.bn12_5(self.conv12_5(x)))                       # batch_size x 16 x 128 x 128
        

        # upsampling13
        x = torch.sigmoid(self.conv13(x))                                   # batch_size x 1 x 128 x 128
        out = self.upscale_13(x)                                        # batch_size x 1 x 256 x 256
        
        return out