import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from .criterion import CrossEntropyCriterion


@LOSSES.register_module()
class SegLoss(nn.Module):
    """SoftmaxLoss.
    """
    def __init__(self,
                 num_classes,
                 loss_weight=1.0,
                 use_point=False):
        super(SegLoss, self).__init__()

        self.criterion = CrossEntropyCriterion(num_classes)

        self.loss_weight = loss_weight

    def forward(self,
                outputs,
                label,
                ignore_index=255,
                ):
        """Forward function."""
        losses = self.criterion(outputs, label, ignore_index)

        return losses