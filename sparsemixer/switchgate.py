import torch
import torch.nn as nn
import torch.nn.functional as F

    
class SwitchGate(nn.Module):
    def __init__(self, num_experts, embed_dim, compute_balance_loss=False, jitter_eps=0.1):
        super(SwitchGate, self).__init__()
        self.num_experts = num_experts
        self.compute_balance_loss = compute_balance_loss
        self.jitter_eps = jitter_eps

    def forward(self, logits):
        if self.training:
            noise = torch.rand_like(logits)
            noise = noise * 2 * self.jitter_eps + 1.0 - self.jitter_eps
            logits = logits * noise
            
        p = logits.softmax(dim=-1)
        sample = torch.argmax(p, dim=-1)
        
        balance_loss = 0.0
        if self.compute_balance_loss:
            num_tokens = F.one_hot(sample, self.num_experts).gt(0).sum(0)
            f = num_tokens / (num_tokens.sum(0, keepdim=True) + 1e-6)
            pmean = p.view(-1, self.num_experts).mean(0) 
            balance_loss = self.num_experts * torch.sum(pmean * f)
        
        multiplier = p.gather(dim=-1, index=sample.unsqueeze(1))
        return sample, multiplier, balance_loss