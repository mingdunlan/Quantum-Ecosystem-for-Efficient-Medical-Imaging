#Alexnet module architecture

import numpy as np
import torch 
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, number_of_classes=100):
        super(Alexnet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.cnn2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.cnn3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.cnn4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.cnn5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        self.fc1 = nn.Linear(in_features=9216, out_features=4096, bias=True)

        self.drop1 = nn.Dropout(0.5, inplace=False)

        self.drop2 = nn.Dropout(0.5, inplace=False)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=100, bias=True)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = self.cnn3(x)
        x = torch.relu(x)
        
        x = self.cnn4(x)
        x = torch.relu(x)
        
        x = self.cnn5(x)
        x = torch.relu(x)
        x = self.maxpool3(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x
