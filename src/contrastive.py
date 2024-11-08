import torch
from torch import nn

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        args.s_dim: the dimension of student's feature
        args.t_dim: the dimension of teacher's feature
        args.feat_dim: the dimension of the projection space
        args.nce_k: number of negatives paired with each positive
        args.nce_t: the temperature
        args.nce_m: the momentum for updating the memory buffer
        args.n_data: the number of samples in the training set, therefor the memory buffer is: args.n_data x args.feat_dim
    """
    def __init__(self, args):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(args.s_dim, args.feat_dim).cuda()
        self.embed_t = Embed(args.t_dim, args.feat_dim).cuda()
        self.contrast = ContrastLoss().cuda()

    def forward(self, f_s, f_t):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s) # embed是线性层+二范式归一化, embed之后的学生特征
        f_t = self.embed_t(f_t) # embed之后的教师特征
        
        loss = self.contrast(f_s, f_t)
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, f_s, f_t):
        bsz = f_s.shape[0]  # batch_size

        mul = torch.matmul(f_s, f_t.T)
        
        # # positive samples of crd
        # m = 49
        # Pn = 1 / float(50)
        # mul = torch.exp(mul)
        # log_pos = torch.div(mul, mul.add(m * Pn + eps)).log_()  # eps = 1e-7
        # loss = - (torch.diag(log_pos).view(-1,1).sum(0)) / bsz

        # inner product of positive samples
        loss = - (torch.diag(mul).view(-1,1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out