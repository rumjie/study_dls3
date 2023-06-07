# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Variable, Model

""" example 1
model = Layer()
model.l1 = L.Linear(5)  # 출력 크기 지정
model.l2 = L.Linear(3)


# 추론
def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


# 모든 매개변수에 접근
for p in model.params():
    print(p)

# 모든 매개변수의 기울기 재설정
model.cleargrad()
"""

"""example 2 - import error
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


x = Variable(np.random.rand(5, 10), name="x")
model = TwoLayerNet(100, 10)
model.plot(x)
"""

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10


# model
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)

# train
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_sqaured_error(y, y_pred)

    model.cleargrad()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 10000 == 0:
        print(loss)
