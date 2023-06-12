# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dezero
from dezero import datasets, optimizers

import math
import numpy as np
import dezero.functions as F
from dezero.models import MLP

# x, t = datasets.get_spiral(train=True)
# print(x.shape)
# print(t.shape)
# print(x[10], t[10])
"""(300, 2)
(300,)
[0. 0.] 0"""

# hyper parameter set
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# data read, model and optimizer generation
x, t = datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 데이터셋 순서 randomized
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 미니 배치 생성
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)  # not simple
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # epoch마다 학습 경과 출력
    avg_loss = sum_loss / data_size
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))
