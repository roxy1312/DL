import torch
from torch import nn
from torchsummary import summary

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.linear1= nn.Linear(100,200)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(200,10)
        self.softmax= nn.Softmax()

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x
    
model=Model()

print(model)

input_size = (50,100)
summary(model, input_size=input_size)