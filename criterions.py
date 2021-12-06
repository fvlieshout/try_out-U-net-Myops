import torch
from torchgeometry.losses import DiceLoss
import torch.nn.functional as f

class Diceloss(torch.nn.Module):
    def init(self):
        super(DiceLoss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class L1loss(torch.nn.Module):
    def init(self):
        super(L1loss, self).init()
    def forward(self, pred, target):
       return f.l1_loss(pred, target).sum()