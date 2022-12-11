import os
import torch
from torch.utils.tensorboard import SummaryWriter


def sequence_mask(lengths: torch.Tensor, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0,max_len,device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size, max_len)
    .ge(lengths.unsqueeze(1)))
    

# https://blog.csdn.net/weixin_53598445/article/details/121301078    
class DefaultTBWriter:
    def __init__(self, path = './log') -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        self.writer = SummaryWriter(path)
        self.step = {}
        
    # 添加单条曲线
    def add_scalar(self, tag, scalar_value, new_style=True):
        self.writer.add_scalar(tag=tag,
                               scalar_value=scalar_value,
                               global_step=self.step.get(tag, 1),
                               new_style=new_style)
        self.step[tag] = self.step(tag, 1) + 1
    