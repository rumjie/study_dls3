# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dezero
from dezero import datasets, optimizers

import math
import numpy as np
import dezero.functions as F
from dezero.models import MLP

# x, t = datasets.get_spiral(train=True)
# print(x.shape)
# print(t.shape)
# print(x[10], t[10])
"""(300, 2)
(300,)
[0. 0.] 0"""

# hyper parameter set
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# data read, model and optimizer generation
x, t = datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)
