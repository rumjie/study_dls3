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
from dezero.datasets import Spiral
from dezero import DataLoader

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasetse.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:  # 훈련용 미니배치
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)  # 훈련 데이터의 인식 정확도
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print("epoch:{}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():  # 기울기 불필요 모드
        for x, t in test_loader:  # 테스트용 미니 배치
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # 테스트 데이터의 인식 정확도
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print(
        "test loss: {:.4f}, accuracy: {:.4f}".foramt(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )
