import torch.nn.functional as F
import torch.nn as nn


class CrossEntropy2D(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-1):
        super(CrossEntropy2D, self).__init__()

        self.loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)
