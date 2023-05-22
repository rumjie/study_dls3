# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
gx = x.grad  # 계산 그래프
x.cleargrad()

z = gx**3 + y
z.backward()
print(x.grad)  # result: variable(100.0)
