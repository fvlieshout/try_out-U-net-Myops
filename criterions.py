import torch
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
    def forward(self, pred, target, device=None):
        loss = f.l1_loss(pred, target, reduction='none')
        return loss.sum()

class MSEloss(torch.nn.Module):
    def init(self):
        super(MSEloss, self).init()
    def forward(self, pred, target, device=None):
        # print('pred1 type', pred.type())
        # print('target1 type', target.type())
        loss = f.mse_loss(pred, target, reduction='none')
        return loss.sum()

class weightedMSEloss(torch.nn.Module):
    def init(self):
        super(weightedMSEloss, self).init()
    def forward(self, pred, target, device=None):
        weights = torch.Tensor([[1.0,1.0,1.0,1.0]])
        weights = weights.to(device)
        ymin_pred, ymax_pred, xmin_pred, xmax_pred = pred.squeeze()
        ymin_real, ymax_real, xmin_real, xmax_real = target.squeeze()
        if ymin_real-ymin_pred < 0:
            weights[:,0] = 2.0
        if ymax_pred-ymax_real < 0:
            weights[:,1] = 2.0
        if xmin_real-xmin_pred < 0:
            weights[:,2] = 2.0
        if xmax_pred-xmax_real < 0:
            weights[:,3] = 2.0
        unweighted_loss = f.mse_loss(pred, target, reduction='none')
        weighted_loss = torch.mul(unweighted_loss, weights)
        return weighted_loss.sum()