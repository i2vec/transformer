import onnx
import torch
from transformer import Transformer
from machine_translation.utils import sequence_mask

X = torch.ones((128, 32), dtype=torch.long)
valid_lens = torch.randint(0, 128, (32,)) 
mask = sequence_mask(valid_lens, max_len=128)

model = Transformer(100, 512, 8, 512, 2048, torch.LongTensor([128, 32, 512]), num_layers=1)
# torch.onnx.export(model, (X, X, mask, mask), "transformer.onnx.pb")

model = torch.jit.trace(model, (X, X, mask, mask))
model.save('./transformer.pt')

# model = 'transformer.onnx.pb'
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)
