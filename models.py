import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousMap(nn.Module):

    def __init__(self, dim_source, dim_dest, width):
        super(ContinuousMap, self).__init__()
        #
        self.fc1 = nn.Linear(dim_source, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, dim_dest)
        
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.fc3(x)
        
        return x


class EncoderAttributes(nn.Module):

    def __init__(self, dim_source1, dim_source2, dim_target, width):
        super(EncoderAttributes, self).__init__()
        #
        self.fc1 = nn.Linear(dim_source1, width)
        self.fc2 = nn.Linear(width + dim_source2, width)
        self.fc3 = nn.Linear(width, dim_target)
        
        self.bn1 = nn.BatchNorm1d(width + dim_source2)
        self.bn2 = nn.BatchNorm1d(width)
        
        self.relu = nn.ReLU()

    def forward(self, features1, features2):
        x = self.relu(self.fc1(features1))
  
        x = torch.cat((x, features2), dim=1)
        x = self.bn1(x)
        
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.fc3(x)
        
        return x
    
    
class DecoderAttributes(nn.Module):

    def __init__(self, dim_source, dim_target1, dim_target2, width):
        super(DecoderAttributes, self).__init__()
        #
        self.fc1 = nn.Linear(dim_source, width)
        self.fc2 = nn.Linear(width, width+dim_target1)
        self.fc3 = nn.Linear(width, dim_target2)
        
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        
        self.relu = nn.ReLU()
        
        self.width = width

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.fc2(x)
        out1 = x[:, self.width:]
        x = x[:, :self.width]
        
        x = self.relu(x)
        x = self.bn2(x)
        
        out2 = self.fc3(x)

        return out1, out2
    

class ContinuousMapResidual(nn.Module):

    def __init__(self, dim_source, dim_dest, width):
        super(ContinuousMapResidual, self).__init__()
        #
        self.fc1 = nn.Linear(dim_source, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, dim_dest)
        
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        
        identity = x
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = x + identity
        
        identity = x
        x = self.relu(self.fc3(x))
        x = self.bn3(x)
        x = x + identity
        
        x = self.fc4(x)
        
        return x


'Linear Decoder to compute a baseline'
class LinearDecoderAttributes(nn.Module):

    def __init__(self, dim_source, dim_target1, dim_target2, width):
        super(LinearDecoderAttributes, self).__init__()
        #
        self.fc1 = nn.Linear(dim_source, width + dim_target1)
        self.fc2 = nn.Linear(width, dim_target2)
        self.width = width

    def forward(self, x):
        x = self.fc1(x)

        out1 = x[:, self.width:]
        x = x[:, :self.width]
        out2 = self.fc2(x)
        return out1, out2