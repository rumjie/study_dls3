import numpy as np
from dezero import Function
from dezero import Variable
import math


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


# 함수 정의
def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):  # threshold = 정밀도
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


# 계산
x = Variable(np.array(np.pi / 4))
y1 = sin(x)
y1.backward()

print(y1.data)  # 0.7071067811865475
print(x.grad)  # 0.7071067811865476

x.cleargrad()

y2 = my_sin(x)
y2.backward()

print(y2.data)  # 0.7071064695751781
print(x.grad)  # 0.7071032148228457
