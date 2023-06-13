import numpy as np
from dezero.core import Function, as_variable, Variable, as_array
from dezero import utils


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)  # 분수 함수의 미분 활용
        return gx


class Exp(Function):  # add by rumjie
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def exp(x):
    return Exp()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape  # 변형 목표 형상

    def forward(self, x):
        self.x_shape = x.shape  # 형상 기억
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


# 구현
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = transpose(gy)
        return gx


# 구현
def transpose(x):
    return Transpose()(x)


class Sum(Function):  # step39
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)  # 다음 단계에서 구현
        return gx


# 구현
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):  # step40
    def __init__(self, shape):
        self.shape = shape

    def forawrd(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


# 구현
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):  # step40
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


# 구현
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


def linear_simple(x, W, b=None):  # step43
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None  # t data delete
    return y


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))  # 책에선 그냥 exp
    return y


def linear(x, W, b=None):  # add by rumjie
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None  # t data delete
    return y


class Sigmoid(Function):  # add by rumjie
    def forward(self, x):
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


# 더 나은 구현 방식----
######


def softmax_cross_entropy_simple(x, t):
    x, y = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = np.clip(p, 1e-15, 1.0)  # log(0) 방지
    log_p = np.log(p)  # ..dezero 함수?
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return Variable(as_array(acc))
