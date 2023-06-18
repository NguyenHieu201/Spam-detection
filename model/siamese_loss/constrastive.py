import torch
import torch.nn as nn


class ConstrastiveLoss(nn.Module):
    def __init__(self, m=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m = m
        self.dist_metrics = nn.PairwiseDistance(p=2, keepdim=True)

    def forward(self, embedd1, embedd2, label) -> torch.Tensor:
        distance = self.dist_metrics(embedd1, embedd2)
        label = label.unsqueeze(-1)
        similar = (1 - label) * distance
        disimilar = label * torch.clamp(self.m - distance, min=0)
        L = similar + disimilar
        L = torch.sum(L) / L.shape[0]
        print(distance)
        return L
