import torch
import torch.nn as nn


class STL10Loss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(STL10Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, out, label):
        """
        arguments:
        out     --  model's output
        label   --  ground truth
        """
        loss_val = self.loss(out, label)
        return loss_val
