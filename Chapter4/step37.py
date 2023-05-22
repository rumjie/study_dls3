# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from dezero import Variable

# 39단계의 함수를 사용했기 때문에 실행은 X
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
