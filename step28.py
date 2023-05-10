import numpy as np
from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)
# result: -2.0, 400.0
