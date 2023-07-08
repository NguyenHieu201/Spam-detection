import torch.nn as nn
from online_triplet_loss.losses import batch_hard_triplet_loss

class TripletLoss(nn.Module):
    def __init__(self, m) -> None:
        super(TripletLoss, self).__init__()
        self.online_triplet_loss = batch_hard_triplet_loss
        self.m = m
        
    def forward(self, embedds, labels):
        loss = self.online_triplet_loss(labels, embedds, self.m)
        return loss