'''
    CSE583 PRML Term Project
    Code for Models for ConvNet
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing Model definition used for the project.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    '''
        Convolutional Neural Network model for classification

        Args: 
            input_shape : Shape of input tensor. (batch_size, 78, 13) for this particular use case
            num_classes : Number of classes to classify. 4 in this case corresponding to American, Australian, Indian and British
    
    '''
    def __init__(self, input_shape, num_classes):
        '''
            Constructor initializing the the network. Doesn't return anything but creates various variables for model creation
        '''
        super(ConvNet, self).__init__()

        # Defining the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Defining the fully connected layers
        self.fc1 = nn.Linear(in_features=64*8, out_features=128)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes)
        
    def forward(self, x):
        '''
            The forward function that takes the model layer by layer and runs the input.
            Output: Tensor of shape (batch_size, num_classes)
        '''

        # trannpose input into required shape
        x = x.transpose(1, 2)

        # Applying convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flattern the output
        x = x.view(x.size(0), -1)

        # Applying the FCN layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        # Result for class
        x = self.fc3(x)
        return x