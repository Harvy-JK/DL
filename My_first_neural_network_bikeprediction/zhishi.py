import numpy as np
import sys
import torch

a = torch.randn(4, 4)
a1 = torch.rand(4, 4)
b = torch.mean(a, 1)
print(a,end='\n\n')
print(a1,end='\n\n')
print(b)

biases = torch.randn(10, dtype=torch.double, requires_grad=True)
print(biases)

