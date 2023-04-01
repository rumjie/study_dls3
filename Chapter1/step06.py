import numpy as np 

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None #미분값 gradient 저장

# function class 구현
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input #입력 변수 보관
        return output

    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

# function 클래스를 상속, 입력값을 제곱하는 클래스
class Square(Function):
    def forward(self, x):
        return x ** 2

# exp function
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

# numerical differentitation
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

# result: 4.000000000004

# 합성함수의 미분
def g(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

# 