import torch

def Gline_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the Gline iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        G = torch.clamp((union - 4*length) / (union + 1e-9),min=0.,max=1.)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    G[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9) - G.sum(dim=-1)
    # num = torch.count_nonzero(ovr,dim=1)
    # weights = torch.zeros(ovr.size())
    # def find_first_nonezero(x):
    #     index = torch.arange(x.shape[1]).unsqueeze(0).repeat((x.shape[0],1))
    #     index[x==0] = x.shape[1]
    #     return torch.min(index,dim=1)[0]
    # a = find_first_nonezero(ovr)
    return iou
def Gliou_loss(pred, target, img_w, length=15):
    return (1 - Gline_iou(pred, target, img_w, length)).mean()