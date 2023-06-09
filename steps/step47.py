# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F
import numpy as np

model = MLP((10, 3))


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y  # 브로드캐스트 수행


x = Variable(np.array[[0.2, -0.4]])
y = model(x)
p = softmax1d(y)
print(y)
print(p)
