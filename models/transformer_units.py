import torch
import torch.nn as nn


# def sequence_mask(valid_lens, max_len: int = None) -> torch.Tensor:
#         batch_size = valid_lens.numel()
#         max_len = max_len or valid_lens.max()
#         return (torch.arange(0, max_len, device=valid_lens.device)
#                 .type_as(valid_lens)
#                 .unsqueeze(0)
#                 .expand(max_len, batch_size)
#                 .ge(valid_lens.unsqueeze(1)))



class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, X):
        return self.linear2(self.relu(self.linear1(X)))
    

class AddNorm(nn.Module):
    def __init__(self, normalized_shape: torch.Tensor, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.linear(self.dropout(Y) + X)
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, maxlen=10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((maxlen, 1, hidden_size))
        X = torch.arange(maxlen, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)
        self.P[:, :, 0::2] = torch.sin(X).unsqueeze(1)
        self.P[:, :, 1::2] = torch.cos(X).unsqueeze(1)
        
    def forward(self, X):
        X = X + self.P[: X.shape[0], :, :].to(X.device)
        return self.dropout(X)  
        


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int = 512, num_heads: int = 8, 
                 normalized_shape: torch.Tensor = torch.Tensor([128, 32, 512]), 
                 ffn_input_dim: int = 512, ffn_hidden_dim = 2048, 
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.position_wise_ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, embed_dim)
        self.add_norm2 = AddNorm(normalized_shape, dropout)
    
    def forward(self, X, key_padding_mask):
        # print(X.shape)
        attn_output, _ = self.multihead_attn(X, X, X, 
                                             key_padding_mask = key_padding_mask)
        X2 = self.add_norm1(X, attn_output)
        return self.add_norm2(X2, self.position_wise_ffn(X2))
        

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int = 512, num_heads: int = 8,
                 normalized_shape: torch.Tensor = torch.Tensor([128, 32, 512]), 
                 ffn_input_dim: int = 512, ffn_hidden_dim: int = 2048, 
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.add_norm2 = AddNorm(normalized_shape, dropout)
        self.position_wise_ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, embed_dim)
        self.add_norm3 = AddNorm(normalized_shape, dropout)
    
    def forward(self, X, encoder_state, key_padding_mask, attn_mask):
        encoder_output, encoder_key_padding_mask = encoder_state
        attn_output, _ = self.multihead_attn1(X, X, X, 
                                              key_padding_mask = key_padding_mask,
                                              attn_mask = attn_mask)
        X2 = self.add_norm1(X, attn_output)
        attn_output, _ = self.multihead_attn2(X2, encoder_output, encoder_output, 
                                              key_padding_mask = encoder_key_padding_mask)
        X3 = self.add_norm2(X2, attn_output)
        return self.add_norm3(X3, self.position_wise_ffn(X3))