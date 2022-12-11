import torch
from models.transformer import Transformer
from utils import sequence_mask
from data.data_proc import enlang, zhlang, DefaultDataset

a = torch.ones((3, 3))
print(a)
print(torch.LongTensor([0]).expand(a.shape[0]))
print(torch.cat((torch.LongTensor([0]).unsqueeze(0).expand(1, a.shape[1]), a[: -1, :]), dim=0))


# print(enlang.num_words, zhlang.num_words)
# train_dataset = DefaultDataset('data/en-zh/train.en', 'data/en-zh/train.zh')
# print(enlang.num_words, zhlang.num_words)
# test_dataset = DefaultDataset('data/en-zh/test.en', 'data/en-zh/test.zh')
# print(enlang.num_words, zhlang.num_words)
# a, b = test_dataset[4]
# print(a)
# print(b)


# test = Transformer(100, 512, 8,  512, 2048, torch.LongTensor([32, 8, 512]), max_len=32)

# X = torch.ones((32, 8), dtype=torch.long)
# valid_lens = torch.randint(0, 32, (8,)) 
# print(valid_lens)
# valid_lens = sequence_mask(valid_lens, max_len=32)
# Y = X
# print(valid_lens)
# print(test(X, Y, valid_lens, valid_lens))