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

class WeightedMSEloss(torch.nn.Module):
    def init(self):
        super(WeightedMSEloss, self).init()
    def forward(self, pred, target, device=None):
        weights = torch.Tensor([[1.0,1.0,1.0,1.0]])
        weights = weights.to(device)
        ymin_pred, ymax_pred, xmin_pred, xmax_pred = pred.squeeze()
        ymin_real, ymax_real, xmin_real, xmax_real = target.squeeze()
        if ymin_real < ymin_pred:
            weights[:,0] = 2.0
        if ymax_pred < ymax_real:
            weights[:,1] = 2.0
        if xmin_real < xmin_pred:
            weights[:,2] = 2.0
        if xmax_pred < xmax_real:
            weights[:,3] = 2.0
        unweighted_loss = f.mse_loss(pred, target, reduction='none')
        weighted_loss = torch.mul(unweighted_loss, weights)
        return weighted_loss.sum()
    
class IoUloss(torch.nn.Module):
    def init(self, loss=True, generalized=False):
        super().__init__()
        self.generalized = generalized
        self.loss = loss

    def forward(self, pred, target, device=None):
        pred = pred.squeeze()
        target = target.squeeze()

        #make sure x1 < x2 and y1 < y2
        p_x1 = min(pred[2], pred[3])
        p_x2 = max(pred[2], pred[3])
        p_y1 = min(pred[0], pred[1])
        p_y2 = max(pred[0], pred[1])

        g_y1, g_y2, g_x1, g_x2 = target

        #calculate area of bounding boxes
        area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
        area_p = (p_x2 - p_x1) * (p_y2 - p_y1)

        #calculate intersection
        i_x1 = max(p_x1, g_x1)
        i_x2 = min(p_x2, g_x2)
        i_y1 = max(p_y1, g_y1)
        i_y2 = min(p_y2, g_y2)

        if i_x2 > i_x1 and i_y2 > i_y1:
            area_i = (i_x2 - i_x1) * (i_y2 - i_y1)
        else:
            area_i = 0
        
        # Finding the coordinates of smallest enclosing box c
        c_x1 = min(p_x1, g_x1)
        c_x2 = max(p_x2, g_x2)
        c_y1 = min(p_y1, g_y1)
        c_y2 = max(p_y2, g_y2)

        area_c = (c_x2 - c_x1) * (c_y2 - c_y1)

        union_area = area_p + area_g - area_i

        IoU = area_i / union_area
        GIoU = IoU - (area_c - union_area)/area_c

        if self.generalized:
            if self.loss:
                return 1 - GIoU
            else:
                return GIoU
        else:
            if self.loss:
                return 1-IoU
            else:
                return IoU