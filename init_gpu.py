
import torch

# set devices & gpu numbers:
device = torch.device("cpu")
mcuda = 0 #if multi_cuda_gpu
if(torch.cuda.is_available()):
  device = torch.device("cuda")
  if (torch.cuda.device_count() > 1):
    mcuda = 1 
else:
  if(torch.backends.mps.is_available()):
    device = torch.device("mps")
print("mcuda: ", mcuda)
print("device:", device)


# insert the code:

if(mcuda):
  net = torch.nn.DataParallel(net)
  #net=torch.nn.DataParallel(net, device_ids=[0, 1, 2])
net.to(device)


# attend:
data.to(device)
    



