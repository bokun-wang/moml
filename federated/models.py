import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    #classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # init.kaiming_normal_(m.weight)
        init.xavier_normal_(m.weight) 


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=[80, 60], in_planes=32*32*3, num_classes=10, last_activation=None, pretrained=False):
        super(NeuralNetwork, self).__init__()
        self.in_planes = in_planes
        self.hidden_size = hidden_size
        assert len(hidden_size) == 2, 'Only support 2 layers!'
        self.dense_layer1 = nn.Linear(in_planes, hidden_size[0], bias=False)
        self.dense_layer2 = nn.Linear(hidden_size[0], hidden_size[1], bias=False)
        self.linear = nn.Linear(hidden_size[1], num_classes, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.apply(_weights_init)
        
        # self.vars = self.parameters

    def forward(self, x):
        x = self.dense_layer1(x)
        x = F.elu(x)
        x = self.dense_layer2(x)
        x = F.elu(x)
        x = self.linear(x)
        if self.last_activation == 'sigmoid':
            x = self.sigmoid(x)
        elif self.last_activation==None:
            x = x                
        else:
            x = self.sigmoid(x)
        return x

    # def parameters(self):
    #     return self.vars


if __name__ == '__main__':
    inputs = torch.rand(5, 784)
    model = NeuralNetwork(in_planes=28*28, hidden_size=[80, 60], num_classes=10)
    outputs = model(inputs)
    
    # print (list(model.parameters()) )