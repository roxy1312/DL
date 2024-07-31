import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#basic implementation of ResNet-34


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


        '''
        if input channels is equal to output channels then we use identity block(adding the input to the output as we can do this when inp_ch==out_ch),
        when input_channels is not equal to output channels we make the input dimension equal to the output dimensions by passing it through a 1x1 kernel conv layer.
        '''
        self.shortcut = nn.Sequential() 
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) #add the shortcut to the output of the second convolution
        out = F.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        self.Conv2_x = self.make_layer(64, 64, blocks=3, stride=1) 
        self.Conv3_x = self.make_layer(64, 128, blocks=4, stride=2)
        self.Conv4_x = self.make_layer(128, 256, blocks=6, stride=2)
        self.Conv5_x = self.make_layer(256, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000) 
    
       
    def make_layer(self, in_channels, out_channels, blocks, stride): #sequence of residual blocks for a given layer of the architecture
        layers = []
        layers.append(Block(in_channels, out_channels, stride)) #creates the first residual block with the specified number of input channels, output channels, and stride
        
        for i in range(1, blocks): #loops through the number of remaining blocks(as thes first block has been added)
            layers.append(Block(out_channels, out_channels))#adds the additional blocks
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.Conv2_x(x)
        x = self.Conv3_x(x)
        x = self.Conv4_x(x)
        x = self.Conv5_x(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

model = ResNet34()
input_size = (3,224,224)
summary(model, input_size=input_size)