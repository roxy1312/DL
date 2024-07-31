import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#basic implementation of alexnet

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=4)
        self.pool1=nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.conv2=nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=1, padding=2)
        self.pool2=nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.conv3=nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=1, padding=1)
        self.conv4=nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=1, padding=1)
        self.conv5=nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.fc1=nn.Linear(256 * 6 * 6, 4096)  
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,1000)  
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Network()
input_size = (3, 227, 227)
summary(model, input_size=input_size)