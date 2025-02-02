import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * x.shape[-1] ** (-0.5)
        return x / (norm + self.eps) * self.weight

def precompute_rope_frequencies(dim: int, max_position: int, theta: float = 10000.0):
    position = torch.arange(max_position)
    freqs = torch.exp(-math.log(theta) * torch.arange(0, dim, 2).float() / dim)
    freqs = position.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([freqs, freqs], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 2048, theta: float = 10000.0):
        super().__init__()
        # Create position embeddings for the maximum sequence length
        position = torch.arange(max_position)
        # Make sure dim is even
        dim = dim // 2
        freqs = torch.exp(-math.log(theta) * torch.arange(0, dim).float() / dim)
        emb = position.unsqueeze(1) * freqs.unsqueeze(0)  # [max_position, dim/2]
        # Complex rotation
        self.register_buffer("cos", torch.cat([emb.cos(), emb.cos()], dim=-1))  # [max_position, dim]
        self.register_buffer("sin", torch.cat([emb.sin(), emb.sin()], dim=-1))  # [max_position, dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        # q, k shape: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Get the cos and sin values for the current positions
        # position_ids shape: [batch_size, seq_len]
        cos = self.cos[position_ids]  # [batch_size, seq_len, head_dim]
        sin = self.sin[position_ids]  # [batch_size, seq_len, head_dim]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
        sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        
        # Apply rotary embeddings
        q_rot = rotate_half(q)
        k_rot = rotate_half(k)
        
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        
        return q, k

class AttentionBlock(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_length, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        q, k = self.rotary(q, k, position_ids)
        
        # Repeat k,v if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        q = q.transpose(1, 2)  # (batch, num_heads, seq_length, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.o_proj(output)

class FeedForward(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.attention = AttentionBlock(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), position_ids, attention_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device)[None].expand(input_ids.size(0), -1)
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(input_ids.shape[0], 1, 1, input_ids.shape[1])
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('0.0'))
        
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

def create_model(pretrained=False):
    config = SmolLM2Config()
    model = SmolLM2(config)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights loading not implemented yet")
    
    return model

if __name__ == "__main__":
    # Test the model
    config = SmolLM2Config()
    model = SmolLM2(config)
    
    # Create sample input
    batch_size, seq_length = 1, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Output shape: {outputs.shape}")  # Should be [1, 128, 49152]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}") 