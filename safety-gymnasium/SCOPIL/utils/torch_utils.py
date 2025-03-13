import torch as th
import torch.nn as nn


class ExpectileLoss(nn.Module):
    def __init__(self, expectile=0.5, reduction="mean"):
        super(ExpectileLoss, self).__init__()
        self.expectile = expectile
        self.reduction = reduction

    def forward(self, predictions, targets):
        errors = predictions - targets
        weights = th.where(errors > 0, self.expectile, 1 - self.expectile)
        loss = weights * (errors ** 2)
        if self.reduction == "mean":
            return th.mean(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"The specified 'reduction' is not supported: {self.reduction}")

