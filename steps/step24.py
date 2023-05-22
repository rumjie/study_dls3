import numpy as np
from dezero import Variable

# sphere
def sphere(x, y):  # 3차원 공간의 sphere 함수
    z = x**2 + y**2
    return z


# matyas
def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


# Goldstein-Price
def goldstein(x, y):
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


x1 = Variable(np.array(1.0))
y1 = Variable(np.array(1.0))
z1 = sphere(x1, y1)
z1.backward()
print("sphere:", x1.grad, y1.grad)  # result: 2.0, 2.0

x2 = Variable(np.array(1.0))
y2 = Variable(np.array(1.0))
z2 = matyas(x2, y2)
z2.backward()
print("matyas:", x2.grad, y2.grad)  # result: 0.040000000000000036, 0.040000000000000036
# 변수 재사용 시 grad가 더해짐

x3 = Variable(np.array(1.0))
y3 = Variable(np.array(1.0))
z3 = goldstein(x3, y3)
z3.backward()
print("goldstein:", x3.grad, y3.grad)  # result: -5376.0 8064.0
