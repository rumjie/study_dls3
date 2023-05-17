import numpy as np
from dezero import Variable


def f(x):
    y = x**4 - 2 * x**2
    return y


x = Variable(np.array(2.0))

y = f(x)
y.backward(create_graph=True)
print(x.grad)  # variable(24.0)

# 두 번째 역전파
gx = x.grad
x.cleargrad()  # 미분값 재설정
gx.backward()
print(x.grad)  # variable(68.0) == 24+44 -> variable(44.0)
