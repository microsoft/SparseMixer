import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMixerCore(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the balancing strategy used in SparseMixer.
    """
    
    @staticmethod
    def forward(
        ctx, 
        multiplier: torch.Tensor, 
        firstorder_mask: torch.Tensor,
    ):
        firstorder_mask = torch.add(0.5, firstorder_mask, alpha=0.5).type_as(multiplier)
        return multiplier * firstorder_mask # turns [0,1] into [0.5, 1]
    
    @staticmethod
    def backward(
        ctx, 
        grad_at_multiplier: torch.Tensor, 
    ):
        return grad_at_multiplier * 2, None
    
class SparseMixer(nn.Module):
    def __init__(self, num_experts, embed_dim, compute_balance_loss=False, jitter_eps=0.1):
        super(SparseMixer, self).__init__()
        self.num_experts = num_experts
        self.compute_balance_loss = compute_balance_loss
        self.jitter_eps = jitter_eps
        self.embed_dim = embed_dim
        self.register_parameter('omega', torch.nn.Parameter(torch.ones(embed_dim)))

    def forward(self, logits):
        
        # masking out experts that are never sampled by jittering 
        with torch.no_grad():
            mask_logits_threshold, max_ind = logits.max(dim=-1, keepdim=True)
            factor = logits.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - logits) / factor
            ) > (2 * self.jitter_eps)
        logits = logits.masked_fill_(mask_logits_threshold, float('-inf'))
        
        p = logits.softmax(dim=-1)
        if self.training:
            sample = (
                logits - torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
            
            multiplier = p.gather(dim=1, index=sample)
            
            mask_for_firstorder = torch.logical_or(
                sample == max_ind,
                torch.rand_like(multiplier) > 0.5
            ) # compute the mask for applying the first-order method
            multiplier = SparseMixerCore.apply(multiplier, mask_for_firstorder) # balance mid-point and euler 
        else:
            sample = max_ind
            multiplier = p.gather(dim=1, index=sample)
        
        multiplier = multiplier * self.omega 
        balance_loss = 0.0
        if self.compute_balance_loss:
            num_tokens = F.one_hot(sample.squeeze(-1), self.num_experts).gt(0).sum(0)
            f = num_tokens / (num_tokens.sum(0, keepdim=True) + 1e-6)
            pmean = p.view(-1, self.num_experts).mean(0) 
            balance_loss = self.num_experts * torch.sum(pmean * f)
        
        return sample, multiplier, balance_loss
        # return sample, [multiplier], balance_loss
