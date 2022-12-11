import torch


class DefaultConfig:
    def __init__(self, 
                 vocab_size, 
                 embed_dim = 512,
                 num_heads = 8,
                 num_layers = 6,
                 max_len = 64,
                 ffn_input_dim = 512,
                 ffn_hidden_dim = 2048,
                 batch_size = 32,
                 lr = 0.01, 
                 dropout = 0.1, 
                 epochs = 50, 
                 ) -> None:
        self.device = 'cpu'
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ffn_input_dim = ffn_input_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.normalized_shape = torch.LongTensor([max_len, batch_size, embed_dim])
        
        if torch.cuda.is_available():
            self.device = 'cuda'