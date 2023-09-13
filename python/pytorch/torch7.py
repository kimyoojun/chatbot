import torch.nn as nn
import torch


class SingleLayer(nn.MOdule):
    def __init__(self, inputs):
        super().__init__()
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x 
    
def main():
    # model = nn.Linear(in_features=1, out_features=1, bias = True)
    # model = nn.Linear(1, 1) # 4번과 같은 라인.
    # print(model.weight)
    # print(model.bias)
    model = MLP()

    print(list(model.children()))
    

if __name__ == '__main__':
