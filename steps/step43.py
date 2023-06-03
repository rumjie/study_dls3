# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2**np.pi * x) + np.random.rand(100, 1)  # data 생성

# 가중치 초기화
I, H, O = 1, 10, 1  # 입력층 차원 수 / 은닉층 차원 수 / 출력층 차원수
W1 = Variable(0.01 * np.random.rand(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, 0))
b2 = Variable(np.zeros(0))


# 신경망 추론
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)  # 책: sigmoid()
    y = F.linear(y, W2, b2)  # 책: linear()
    return y


lr = 0.2
iters = 10000

# 신경망 학습, 매개변수 갱신
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:  # 1000회마다 출력
        print(loss)
