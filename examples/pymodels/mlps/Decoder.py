import torch.nn as nn

class Decoder(nn.Module):
  def __init__(self, input_size):
    super(Decoder, self).__init__()
    self.decoder = 
    nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    return self.decoder(x)

import torch
import time

def randTensor(*shape, dtype=torch.float32):
  if dtype.is_floating_point:
    return torch.rand(*shape, dtype=dtype) * 2 - 1
  elif dtype == torch.bool:
    return torch.randint(0, 2, shape, dtype=dtype)
  else:
    if dtype.is_signed:
      return torch.randint(-10, 10, shape, dtype=dtype)
    else:
      return torch.randint(0, 10, shape, dtype=dtype)

def saveWeights(model, outPath):
  print("Saving weights to", outPath)
  # save weights as binary files
  idx = 0
  for name, param in model.named_parameters():
    with open(outPath + f"weight_{idx}.bin", "wb") as f:
      f.write(param.detach().numpy().tobytes())
    idx += 1

def generateGoldenResults(model, inputs, outPath):
  print("Saving golden results to", outPath)
  # save inputs as binary files
  for i, input in enumerate(inputs):
    with open(outPath + f"input_{i}.bin", "wb") as f:
      f.write(input.detach().numpy().tobytes())
  # measure inference time
  model.train(False)
  model.eval()
  start = time.time()
  outputs = model(*inputs)
  end = time.time()
  # if outputs is a tuple, store each tensor as a separate binary file
  if isinstance(outputs, tuple):
    for idx, output in enumerate(outputs):
      # store torch tensor as binary file
      with open(outPath + f"output_{idx}.bin", "wb") as f:
        f.write(output.detach().numpy().tobytes())
  else:
    # store torch tensor as binary file
    with open(outPath + "output_0.bin", "wb") as f:
      f.write(outputs.detach().numpy().tobytes())
  return end - start


random_input = randTensor(1, 1024, dtype=torch.float32)
decoder = Decoder(1024)
decoder.eval()
output = decoder(random_input)
generateGoldenResults(decoder, (random_input, ), "./data/")
saveWeights(decoder, "./data/")

