import torch
from hstu import GenRec

if __name__ == '__main__':

    dim = 64
    model = GenRec(d_model=dim, num_heads=2,num_layers=3)
    x = torch.rand(512, 2000, dim)
    model(x)
