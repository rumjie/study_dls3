# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import dezero
from dezero import datasets, optimizers

import time
import math
import numpy as np
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
from dezero import DataLoader
from dezero import optimizers

max_epoch = 5
batch_size = 100

train_set = dezero.datsets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size):
model = MLP((1000,10))
optimizers = optimizers.SGD().setup(model)

# GPU 모드
if dezero.cuda_gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss =0

    for x, t in train_loader:
        y=model(x)
        loss = F.softmax_cross_entropy_simple(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data)*len(t)
    
    elapsed_time = time.time()-start
    print('epoch:{}, loss: {:.f4}, time: {:.4f}[sec]'.format(epoch+1, sum_loss/len(train_set), elapsed_time))