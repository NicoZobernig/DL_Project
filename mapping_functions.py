
import torch.nn as nn
import torch.nn.functional as F


class SemanticToVisual(nn.Module):

    def __init__(self, semantic_dim, visual_dim):
        super(SemanticToVisual, self).__init__()
        #
        self.fc1 = nn.Linear(semantic_dim, 3 * semantic_dim)
        self.fc2 = nn.Linear(3 * semantic_dim, 2*visual_dim)
        self.fc3 = nn.Linear(2*visual_dim, visual_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class VisualToSemantic(nn.Module):

    def __init__(self, semantic_dim, visual_dim):
        super(VisualToSemantic, self).__init__()

        self.fc1 = nn.Linear(visual_dim, 2*visual_dim)
        self.fc2 = nn.Linear(2*visual_dim, 3 * semantic_dim)
        self.fc3 = nn.Linear(3 * semantic_dim, semantic_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
