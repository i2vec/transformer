import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_units import TransformerEncoderBlock, PositionalEncoding, TransformerDecoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 vocab_size, embed_dim, num_heads, 
                 normalized_shape,
                 ffn_input_dim, ffn_hidden_dim,
                  
                 dropout: float = 0.1, num_layers = 6) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'block{i}', 
                            TransformerEncoderBlock(embed_dim, 
                                                    num_heads, 
                                                    normalized_shape, 
                                                    ffn_input_dim, 
                                                    ffn_hidden_dim, 
                                                    dropout))
    
    def forward(self, X: torch.Tensor, key_padding_mask):
        # print(f'X shape: {X.shape}')
        # print(f'valid shape: {valid_lens.shape}')
        output = self.position_encoder(self.embedding(X) * math.sqrt(self.embed_dim))
        for encoder_block in self.blocks:
            output = encoder_block(output, key_padding_mask)
        return output
    

class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim, num_heads,
                 ffn_input_dim, ffn_hidden_dim, 
                 normalized_shape: torch.Tensor,  
                 dropout: float = 0.1, num_layers = 6) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'block{i}',
                                   TransformerDecoderBlock(embed_dim, 
                                                           num_heads,
                                                           normalized_shape, 
                                                           ffn_input_dim, 
                                                           ffn_hidden_dim,
                                                           dropout))
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, X: torch.Tensor, encoder_state, key_padding_mask, attn_mask):
        output = self.position_encoder(self.embedding(X) * math.sqrt(self.embed_dim))
        for decoder_block in self.blocks:
            output = decoder_block(output, encoder_state, key_padding_mask, attn_mask)
        return self.linear(output)
    

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim, num_heads,
                 ffn_input_dim, ffn_hidden_dim,
                 normalized_shape: torch.Tensor,
                 dropout: float = 0.1, num_layers = 6, max_len = 128) -> None:
        super().__init__()
        self.max_len = max_len
        self.encoder = TransformerEncoder(vocab_size, 
                                          embed_dim,
                                          num_heads, 
                                          normalized_shape,
                                          ffn_input_dim,
                                          ffn_hidden_dim,
                                          dropout,
                                          num_layers)
        self.decoder = TransformerDecoder(vocab_size,
                                          embed_dim,
                                          num_heads,
                                          ffn_input_dim,
                                          ffn_hidden_dim,
                                          normalized_shape,
                                          dropout,
                                          num_layers)
    
    def forward(self, src, trg, src_padding_mask, trg_padding_mask):
        
        encoder_output = self.encoder(src, src_padding_mask)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.max_len)
        output = self.decoder(trg, (encoder_output, src_padding_mask), trg_padding_mask, attn_mask)
        return output
        