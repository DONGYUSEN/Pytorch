
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
    

#保存模型
#直接保存模型（参数+图）：
torch.save(net, "model.pkl")

#如果环境没有变化， 读取的时候直接读取：
model = torch.load(args.pretrained_model_path)


#如果运行环境发生变化，就要有一定的变化了——先建立模型，然后加载参数：
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# 建立模型
model = MyModel(args)

if torch.cuda.is_available() and args.use_gpu:
    model = torch.nn.DataParallel(model).cuda()

if not (args.pretrained_model_path is None):
    print('load model from %s ...' % args.pretrained_model_path)
    # 获得模型参数
    model_dict = torch.load(args.pretrained_model_path).module.state_dict()
    # 载入参数
    model.module.load_state_dict(model_dict)
    print('success!')

