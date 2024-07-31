import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

#basic implementation of VGG-16

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        #1st block
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=(3,3),stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels= 64, out_channels=64, kernel_size=(3,3),stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        #output dim -> (112,112,64)

        #2nd block
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),stride=1, padding=1)
        self.Conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #output dim -> (56,56,128)

        
        #3rd block
        self.Conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.Conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.Conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #output dim ->( 28,28,256)


        #4th block
        self.Conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.Conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.Conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #output dim -> (14,14,512)

        #5th block
        self.Conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.Conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.Conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #output dim -> (7,7,512)


        self.fc1=nn.Linear(512 * 7 * 7, 4096)  
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,1000)

    

    def forward(self,x):

        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.pool1(x)

        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = self.pool2(x)

        x = F.relu(self.Conv5(x))
        x = F.relu(self.Conv6(x))
        x = F.relu(self.Conv7(x))
        x = self.pool3(x)

        x = F.relu(self.Conv8(x))
        x = F.relu(self.Conv9(x))
        x = F.relu(self.Conv10(x))
        x = self.pool4(x)

        x = F.relu(self.Conv11(x))
        x = F.relu(self.Conv12(x))
        x = F.relu(self.Conv13(x))
        x = self.pool5(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = VGG()
input_size = (3,224,224)
summary(model, input_size=input_size)