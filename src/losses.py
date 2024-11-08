import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, logit_s, logit_t):
        p_s = F.log_softmax(logit_s, dim=1)
        p_t = F.softmax(logit_t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') / logit_s.shape[0]
        return loss
    


def build_loss(loss_name, **kwargs):
    loss_factory = globals()[loss_name]
    loss = loss_factory(**kwargs)
    return loss