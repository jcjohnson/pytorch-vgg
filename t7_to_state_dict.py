import argparse
import os
import torch
import torchvision.models as models
from torch.utils.serialization import load_lua

"""
Read a .t7 file written by caffemodel_to_t7.lua and convert it to a PyTorch .pth
file containing a state dict for a VGG model.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_t7', required=True)
parser.add_argument('--model_name', required=True)
args = parser.parse_args()


t7_model = load_lua(args.input_t7)['model']

pytorch_model = getattr(models, args.model_name)()
feature_modules = list(pytorch_model.features.modules())
classifier_modules = list(pytorch_model.classifier.modules())
pytorch_modules = feature_modules + classifier_modules

next_pytorch_idx = 0
for i, t7_module in enumerate(t7_model.modules):
  if not hasattr(t7_module, 'weight'):
    continue
  assert hasattr(t7_module, 'bias')
  while not hasattr(pytorch_modules[next_pytorch_idx], 'weight'):
    next_pytorch_idx += 1
  pytorch_module = pytorch_modules[next_pytorch_idx]
  next_pytorch_idx += 1
  assert(t7_module.weight.size() == pytorch_module.weight.size())
  print('Copying data from\n  %r to\n  %r' % (t7_module, pytorch_module))

  pytorch_module.weight.data.copy_(t7_module.weight)
  assert(t7_module.bias.size() == pytorch_module.bias.size())
  pytorch_module.bias.data.copy_(t7_module.bias)

initial_path = '%s.pth' % args.model_name
torch.save(pytorch_model.state_dict(), initial_path)

# This is a really dirty way to get the sha256sum of the file...
output = os.system('sha256sum %s > _hash' % initial_path)
with open('_hash', 'r') as f:
  _hash = next(f)
os.remove('_hash')

final_path = '%s-%s.pth' % (args.model_name, _hash[:8])
os.rename(initial_path, final_path)

