import torch
from machine_translation.utils import sequence_mask
from transformer import Transformer

test = Transformer(100, 512, 8,  512, 2048, torch.LongTensor([32, 8, 512]), max_len=32)

X = torch.ones((32, 8), dtype=torch.long)
valid_lens = torch.randint(0, 32, (8,)) 
print(valid_lens)
valid_lens = sequence_mask(valid_lens, max_len=32)
Y = X
print(valid_lens)
print(test(X, Y, valid_lens, valid_lens))

# a = torch.tensor([[[1, 2],
#                    [1, 2],
#                    [1, 2]],
#                   [[1, 2],
#                    [1, 2],
#                    [1, 2]]])

# b = torch.tensor([[[1, 2],
#                    [1, 2],
#                    [1, 2],
#                    [1, 2]],
#                   [[1, 2],
#                    [1, 2],
#                    [1, 2],
#                    [1, 2]]])
# a = a.permute(1, 0, 2)
# b = b.permute(1, 0, 2)

# print(a.shape)
# print(b.shape)
# c = torch.bmm(a.permute(1, 0, 2), b.permute(1, 0, 2).transpose(1, 2)).permute(1, 0, 2)
# print(c.shape)
# d = torch.einsum('jik,lik->jil', a, b)
# print(d.shape)
# print(c == d)

# a = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [0, 0], [0, 0], [0, 0], [0, 0]],
#                   [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 0], [0, 0], [0, 0]]], dtype=torch.float32)
# b = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [0, 0], [0, 0], [0, 0], [0, 0]],
#                   [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [0, 0], [0, 0], [0, 0]]], dtype=torch.float32)

# # print(mask.dim() == 1)
# mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
#                      [1, 1, 1, 1, 1, 0, 0, 0]])
# # a = a.permute(1, 0, 2)
# # b = b.permute(1, 0, 2)
# # mask = mask.permute(1, 0)
# # mask = mask.unsqueeze(-1).expand(a.shape)
# c = torch.einsum('ibh,jbh->ibj', a, b)
# c = torch.einsum('bih,bjh->bij', a, b)
# print(a)
# print(b)
# print(c)
# print(torch.softmax(c, dim=-1))


# a = torch.tensor([1, 2, 0, 2, 0, 0], dtype=torch.float32)
# print(a.masked_fill(a == 0, -100))
    
    
# a = sequence_mask(torch.tensor([1, 3, 4]), max_len=10)
# print(a)