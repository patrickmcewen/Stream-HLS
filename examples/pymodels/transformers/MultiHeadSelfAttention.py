import torch
import torch.nn as nn
import torch.nn.functional as F

# Export functions to prevent dead code elimination
__all__ = [
    'addf', 'mulf', 'subf', 'divf', 'exp_bb',
    'addf_ctrl_chain', 'mulf_ctrl_chain', 'subf_ctrl_chain', 
    'divf_ctrl_chain', 'exp_bb_ctrl_chain',
    'MultiHeadSelfAttention'
]

def addf(a: float, b: float) -> float:
    return a + b

def mulf(a: float, b: float) -> float:
    return a * b

def subf(a: float, b: float) -> float:
    return a - b

def divf(a: float, b: float) -> float:
    return a / b

def exp_bb(a: float) -> float:
    # Putting exp(a) here causes an unsupported instruction to be generated. This is just a placeholder anyways.
    return a * 2

def addf_ctrl_chain(a: float, b: float) -> float:
    return a + b

def mulf_ctrl_chain(a: float, b: float) -> float:
    return a * b

def subf_ctrl_chain(a: float, b: float) -> float:
    return a - b

def divf_ctrl_chain(a: float, b: float) -> float:
    return a / b

def exp_bb_ctrl_chain(a: float) -> float:
    # Putting exp(a) here causes an unsupported instruction to be generated. This is just a placeholder anyways.
    return a * 2

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(context)
        
        return output
