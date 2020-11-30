import torch

IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if IS_CUDA else 'cpu')
