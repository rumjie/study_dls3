if "__file__" in globals(): #전역 변수 설정 확인
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
from dezero.core_simple import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()
print(x)  # <dezero.core_simple.Variable object at 0x104bd7f10>?
print(y)
print(x.grad)  # 8.0
