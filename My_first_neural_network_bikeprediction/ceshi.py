import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

a = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5]

b = np.arange(len(a))[12::24]

print(np.arange(len(a)))

print(b)
