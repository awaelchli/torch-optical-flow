
import torch 

from model import RAFT

model = RAFT()

img1 = torch.rand(8, 3, 368, 496)
img2 = torch.rand(8, 3, 368, 496)
out = model(img1, img2)