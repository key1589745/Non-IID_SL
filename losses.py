import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict



def get_loss(name, **kwargs):
    if name == 'CE':
        return nn.CrossEntropyLoss()
    elif name == 'CE_p':
        return CrossEntropyPartial(kwargs['labels'])
    elif name == 'UNCE':
        return UnbiasedCrossEntropy(kwargs['labels'])
    elif name == 'focal':
        return Focal(kwargs['labels'])
    else:
        raise NotImplementedError
        

class KL_distill_loss(nn.KLDivLoss):
    
    def __init__(self, size_average=None, reduce=None, reduction = 'mean', log_target = True):
        super().__init__(size_average, reduce, reduction, log_target)

    def forward(self, inputs, targets):
        
        inputs = F.log_softmax(inputs, 1)
        targets = F.log_softmax(targets, 1)

        return super().forward(inputs, targets)    
    
def L2_penalty(model, model_cache, omega=defaultdict(lambda: 1)):
    loss = 0
    params = {n: p.data.cuda() for n, p in model_cache.state_dict().items() if p.requires_grad}
    for n, p in model.state_dict().items():
        if p.requires_grad:
            _loss = omega[n] * (p - params[n]) ** 2
            loss += _loss.sum()
    return loss


def FeSA_loss(old_stats,new_stats):
    
    #forcing mean and variance to match between two distributions
    #other ways might work better, i.g. KL divergence
    r_feature = 0.
    for old, new in zip(old_stats, new_stats):
        #print(new.stats['var'].requires_grad)
        r_feature += torch.norm(old['var'] - new[0], 2) + \
        torch.norm(old['mean'] - new[1], 2)
    
    return r_feature


class UnbiasedCrossEntropy(nn.Module):

    def __init__(self, true_labs, old_cl=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.true_labs = true_labs

    def forward(self, inputs, targets):
        un_labels = [i for i in range(4) if i not in self.true_labs]
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        if 0 in un_labels:
            outputs[:, 0] = torch.logsumexp(inputs[:,un_labels], dim=1) - den  # B, H, W       p(O)          
        else:
            outputs[:, 1] = torch.logsumexp(inputs[:,un_labels], dim=1) - den
            
        outputs[:, self.true_labs] = inputs[:, self.true_labs] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
    
class Focal(nn.Module):

    def __init__(self, true_labs, old_cl=None, reduction='none', ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.true_labs = true_labs
        self.alpha = 1
        self.gamma = 2

    def forward(self, inputs, targets):
        un_labels = [i for i in range(4) if i not in self.true_labs]
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        if 0 in un_labels:
            outputs[:, 0] = torch.logsumexp(inputs[:,un_labels], dim=1) - den  # B, H, W       p(O)          
        else:
            outputs[:, 1] = torch.logsumexp(inputs[:,un_labels], dim=1) - den
            
        outputs[:, self.true_labs] = inputs[:, self.true_labs] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)
        
        loss = torch.mean(self.alpha*(1-outputs)**self.gamma*loss)

        return loss

    
class CrossEntropyPseudo(nn.Module):

    def __init__(self, old_cl=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, true_labs):
        un_labels = [i for i in range(4) if i not in true_labs]
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        if 0 in un_labels:
            outputs[:, 0] = torch.logsumexp(inputs[:,un_labels], dim=1) - den  # B, H, W       p(O)          
        else:
            outputs[:, 1] = torch.logsumexp(inputs[:,un_labels], dim=1) - den
            
        outputs[:, true_labs] = inputs[:, true_labs] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
    
class CrossEntropyUncertainty(nn.Module):

    def __init__(self, true_labs, weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.true_labs = true_labs

    def forward(self, pred, targets):
        
        un_labels = [i for i in range(4) if i not in self.true_labs]
        p = 1 / len(un_labels)
        
        if 0 in self.true_labs:
            un_index = (targets==1)
        else:
            un_index = (targets==0)
        
        targets = F.one_hot(targets.clone(), num_classes=4).permute(0,3,1,2) # B, C, H, W 
        
        for i in un_labels:
            targets[:,i][un_index] = p
            
        pred = F.log_softmax(pred, 1)
      
        loss = - torch.sum(targets * pred) / pred.shape[0]

        return loss


class CrossEntropyPartial(nn.CrossEntropyLoss):
    
    def __init__(self, labels, size_average=None, ignore_index= -100, reduce=None, reduction= 'mean'):
        weight = torch.zeros(4,)
        weight[labels] = 1
        
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        
