import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    From: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class QuadrupletLoss(nn.Module):
    """
       Quadruplet loss
       Takes embeddings of an anchor sample, a positive sample and two negative samples
       """

    def __init__(self, margin):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative1, negative2, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1) - (anchor - negative1).pow(2).sum(1)
        distance_negative = (anchor - positive).pow(2).sum(1) - (negative1 - negative2).pow(2).sum(1)
        losses = F.relu(distance_positive + distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
