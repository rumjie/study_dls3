import numpy as np
from dezero import Variable


def f(x):
    y = x**4 - 2 * x**2
    return y


def gx2(x):
    return 12 * x**2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()  # 1차 미분은 역전파로

    x.data -= x.grad / gx2(x.data)  # 2차 미분은 수동으로 코딩
