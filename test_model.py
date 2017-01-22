import argparse
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.serialization import load_lua

"""
Read a .t7 file created by caffemodel_to_t7.lua and a .pth file created by
t7_to_state_dict.py and make sure the converted PyTorch model computes the same
inputs and outputs stored in the .t7 file.

Test cases are computed on CPU using float32 to prevent potential nondeterminism
in cuDNN.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--t7_file', required=True)
parser.add_argument('--pth_file', required=True)
args = parser.parse_args()


model_name = os.path.splitext(args.pth_file)[0].split('-')[0]

model = getattr(models, model_name)()
model.load_state_dict(torch.load(args.pth_file))
model.float()
model.eval()

test_cases = load_lua(args.t7_file)['tests']
for i, test_case in enumerate(test_cases):
  print('Running test case %d / %d' % (i + 1, len(test_cases)))
  x = Variable(test_case['input'].float(), requires_grad=True)
  expected_y = test_case['output'].float()
  grad_y = test_case['grad_output'].float()
  expected_grad_x = test_case['grad_input'].float()

  y = model(x)
  y_diff = torch.abs(y.data - expected_y).sum()
  assert y_diff == 0, 'y_diff = %f' % y_diff
  y.backward(grad_y)
  grad_x_diff = torch.abs(x.grad.data - expected_grad_x).sum()
  assert grad_x_diff == 0, 'grad_x_diff = %f' % grad_x_diff
  
print('All tests pass!')

