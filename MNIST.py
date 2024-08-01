import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim
import pandas as pd
from tqdm import tqdm


train = pd.read_csv('/home/roxy/Downloads/train.csv')



features = train.drop(columns=['label']).values
labels = train['label'].values

features = features / 255.0 #normalizing 
print("features shape: ",features.shape)
features = features.reshape(-1,1,28,28)
print("features shape: ",features.shape)


features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 11 * 11, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model=Network()
input_size=(1,28,28)
summary(model,input_size=input_size)


learning_rate=3e-4
num_epochs=8
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.view(-1, 1, 28, 28)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")