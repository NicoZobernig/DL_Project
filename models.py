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


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.fc3(x)
        
        return x
    
    
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


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        identity = x
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = x + identity
        
        identity = x
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = x + identity
        
        x = self.fc4(x)
        
        return x
    
    

    