from csv import QUOTE_ALL
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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
        
        return output, K.transpose(1, 2), V.transpose(1, 2)


class MultiHeadSelfAttentionKV(nn.Module):
    """
    Multi-Head Self-Attention with KV caching support.
    
    This class provides separate methods for prefill (processing entire sequence)
    and decode (processing single tokens with cached K/V) phases.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttentionKV, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode phase: Process single token using cached K/V.
        
        Args:
            x: Input tensor of shape (batch_size, 1, embed_dim) - single new token
            k_cache: Cached keys of shape (batch_size, cached_len, num_heads, head_dim)
            v_cache: Cached values of shape (batch_size, cached_len, num_heads, head_dim)
            is_causal: Whether to apply causal masking
            
        Returns:
            Tuple of (output, new_k_cache, new_v_cache) where:
            - output: Attention output of shape (batch_size, 1, embed_dim)
            - new_k_cache: Updated keys cache of shape (batch_size, cached_len + 1, num_heads, head_dim)
            - new_v_cache: Updated values cache of shape (batch_size, cached_len + 1, num_heads, head_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        assert seq_len == 1, "Decode should process only one token at a time"
        
        # Project new token to Q, K, V
        Q = self.query(x)  # (batch_size, 1, embed_dim)
        K_new = self.key(x)    # (batch_size, 1, embed_dim)
        V_new = self.value(x)  # (batch_size, 1, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        K_new = K_new.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        V_new = V_new.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        
        # Concatenate cached K/V with new token's K/V
        # k_cache: (batch, cached_len, num_heads, head_dim) -> transpose to (batch, num_heads, cached_len, head_dim)
        k_cache_transposed = k_cache.transpose(1, 2)  # (batch, num_heads, cached_len, head_dim)
        v_cache_transposed = v_cache.transpose(1, 2)  # (batch, num_heads, cached_len, head_dim)
        
        # NOTE: this is a workaround to avoid bufferization dialect. I instead just increased the sequence dimension of K_cache and V_cache by 1.
        #K = torch.cat([k_cache_transposed, K_new], dim=2)  # (batch, num_heads, cached_len + 1, head_dim)
        #V = torch.cat([v_cache_transposed, V_new], dim=2)  # (batch, num_heads, cached_len + 1, head_dim)
        K = k_cache_transposed
        V = v_cache_transposed

        # Compute attention scores: Q @ K^T
        # Q: (batch, num_heads, 1, head_dim)
        # K: (batch, num_heads, cached_len + 1, head_dim)
        # scores: (batch, num_heads, 1, cached_len + 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)  # (batch, num_heads, 1, head_dim)
        
        # Reshape back: (batch, num_heads, 1, head_dim) -> (batch, 1, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        output = self.out(context)
        
        # Update cache: concatenate new K/V to cached K/V
        # Transpose back to (batch, cached_len + 1, num_heads, head_dim) for storage
        new_k_cache = K.transpose(1, 2)  # (batch, cached_len + 1, num_heads, head_dim)
        new_v_cache = V.transpose(1, 2)  # (batch, cached_len + 1, num_heads, head_dim)
        
        return output, new_k_cache, new_v_cache

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x):
        x, k, v = self.attention(x)
        x = self.feedforward(x)
        return x, k, v

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.embed = nn.Embedding(num_tokens, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, num_tokens)

    def forward(self, x):
        #x = self.embed(x)
        #x = self.norm(x)
        for layer in self.layers:
            x, k, v = layer(x)
        #x = self.out(x)
        return x, k, v

class TransformerLayerKV(nn.Module):
    """
    A single transformer layer that combines MultiHeadSelfAttentionKV and FeedForward.
    This custom layer is needed because nn.Sequential can't handle multiple inputs/outputs.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayerKV, self).__init__()
        self.attention = MultiHeadSelfAttentionKV(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
    
    def forward(self, x, k_cache, v_cache):
        # Attention with KV cache
        x, new_k_cache, new_v_cache = self.attention(x, k_cache, v_cache)
        # Feedforward (only takes x)
        x = self.feedforward(x)
        return x, new_k_cache, new_v_cache

class TransformerKV(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout=0.1):
        super(TransformerKV, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.embed = nn.Embedding(num_tokens, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayerKV(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, num_tokens)

    def forward(self, x, k_cache, v_cache): # assume x is already embedded
        #x = self.embed(x)
        #x = self.norm(x)
        for layer in self.layers:
            x, new_k_cache, new_v_cache = layer(x, k_cache, v_cache)
            k_cache = new_k_cache
            v_cache = new_v_cache
        #x = self.out(x)
        return x, k_cache, v_cache

class Inference(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout=0.1):
        super(Inference, self).__init__()
        self.transformer = Transformer(embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout)
        self.transformer_kv = TransformerKV(embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout)

    def forward(self, x, num_decode_steps=2): # runs 1 prefill and 2 decodes
        x, k, v = self.transformer(x)
        print(f"x shape: {x.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        x = x[:, -1:]
        for _ in range(num_decode_steps):
            x, k, v = self.transformer_kv(x, k, v)
        return x