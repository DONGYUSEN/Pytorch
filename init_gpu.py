
import torch

# set devices & gpu numbers:
device = 'cpu'torch.device("cpu")
mcuda = 0
if(torch.cuda.is_available()):
  device = torch.device("cuda")
  if (torch.cuda.device_count() > 1):
    mcuda = 1 
else:
  if(torch.backends.mps.is_available()):
    device = torch.device("mps")
print("mcuda: ", mcuda)
print("device:", device)

