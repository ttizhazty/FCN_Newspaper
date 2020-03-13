import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractNet(nn.Module):
    """
    This network using extract features from the network
    """
    def __init__(self):
        """
        define a convolutional neural network to generate feature map
        
        Args:
            Hard code here, TODO: config network with config file
        """
        super(FeatureExtractNet, self).__init__()
        
        # conv1 downsample=1/2
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv2 downsample=1/4
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv3 downsample=1/8
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn3_2 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv4 downsample=1/16
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.conv4_1 = nn.Conv2d(16, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn4_2 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv5 downsample=1/32
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.conv5_1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.conv5_2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv6 downsample=1/64
        self.dropout6 = nn.Dropout2d(p=0.3)
        self.conv6_1 = nn.Conv2d(128, 128, kernel_size = 5, stride = 1, padding = 2) # different with paper
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn6_2 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # conv7 
        self.dropout7 = nn.Dropout2d(p=0.3)
        self.conv7_1 = nn.Conv2d(256, 256, kernel_size = 5, stride = 1, padding = 2) # different with paper
        self.bn7_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            x(Variable): contains a batch of images, of dimension batch_size x channels x H x W .

        Returns:
            out(list): list of feature maps for each conv step 
            
        Note the dimensions after each step are provided
        """
        #input                                            batch_size x 2 x 256 x 256

        # block1
        x = self.dropout1(x)                            # batch_size x 2 x 256 x 256
        x = F.relu(self.bn1_1(self.conv1_1(x)))         # batch_size x 32 x 256 x 256
        x = F.relu(self.bn1_2(self.conv1_2(x)))         # batch_size x 16 x 256 x 256
        x = self.pool1(x)                               # batch_size x 16 x 128 x 128
        pool1 = x

        # block2
        x = self.dropout2(x)                            # batch_size x 16 x 128 x 128
        x = F.relu(self.bn2_1(self.conv2_1(x)))         # batch_size x 16 x 128 x 128
        x = F.relu(self.bn2_2(self.conv2_2(x)))         # batch_size x 16 x 128 x 128
        x = self.pool2(x)                               # batch_size x 16 x 64 x 64
        pool2 = x

        # block3
        x = self.dropout3(x)                            # batch_size x 16 x 64 x 64
        x = F.relu(self.bn3_1(self.conv3_1(x)))         # batch_size x 16 x 64 x 64
        x = F.relu(self.bn3_2(self.conv3_2(x)))         # batch_size x 16 x 64 x 64
        x = self.pool3(x)                               # batch_size x 16 x 32 x 32
        pool3 = x

        # block4
        x = self.dropout4(x)                            # batch_size x 16 x 32 x 32
        x = F.relu(self.bn4_1(self.conv4_1(x)))         # batch_size x 64 x 32 x 32
        x = F.relu(self.bn4_2(self.conv4_2(x)))         # batch_size x 64 x 32 x 32
        x = self.pool4(x)                               # batch_size x 64 x 16 x 16
        pool4 = x

        # block5
        x = self.dropout5(x)                            # batch_size x 64 x 16 x 16
        x = F.relu(self.bn5_1(self.conv5_1(x)))         # batch_size x 64 x 16 x 16
        x = F.relu(self.bn5_2(self.conv5_2(x)))         # batch_size x 128 x 16 x 16
        x = self.pool5(x)                               # batch_size x 128 x 8 x 8
        pool5 = x

        # block6
        x = self.dropout6(x)                            # batch_size x 128 x 8 x 8
        x = F.relu(self.bn6_1(self.conv6_1(x)))         # batch_size x 128 x 8 x 8
        x = F.relu(self.bn6_2(self.conv6_2(x)))         # batch_size x 256 x 8 x 8
        x = self.pool6(x)                               # batch_size x 256 x 4 x 4
        pool6 = x

        # block1
        x = self.dropout7(x)                            # batch_size x 256 x 4 x 4
        x = F.relu(self.bn7_1(self.conv7_1(x)))         # batch_size x 256 x 4 x 4
        pool7 = x

        return [pool1, pool2, pool3, pool4, pool5, pool6, pool7]