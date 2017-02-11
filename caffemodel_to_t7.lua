require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

--[[
Use loadcaffe to load a Caffe model and store it in a .t7 file.

We also run some random data through the model and store it in the .t7 file so
we can make sure that the model still computes the same thing once we convert
it to PyTorch. Test cases are computed on CPU using float.
]]--


local cmd = torch.CmdLine()
cmd:option('-input_caffemodel', '')
cmd:option('-input_prototxt', '')
cmd:option('-output_t7', '')
cmd:option('-num_tests', 10)

local opt = cmd:parse(arg)
local model = loadcaffe.load(opt.input_prototxt, opt.input_caffemodel, 'nn')
model:remove() -- Remove the softmax at the end
assert(torch.isTypeOf(model:get(#model), nn.Linear))
model:evaluate()
model:float()

local tests = {}
for i = 1, opt.num_tests do
  print(string.format('Making test case %d', i))
  local input = torch.randn(1, 3, 224, 224):float()
  local output = model:forward(input):clone()
  local grad_output = torch.randn(#output):float()
  local grad_input = model:backward(input, grad_output):clone()
  table.insert(tests, {
    input=input,
    output=output,
    grad_output=grad_output,
    grad_input=grad_input,
  })
end

torch.save(opt.output_t7, {
  model=model,
  tests=tests,
})

