import numpy as np 

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 미분값 gradient 저장

# function class 구현
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input # 입력 변수 보관
        return output

    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

# function 클래스를 상속, 입력값을 제곱하는 클래스
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy): #역전파를 담당하는 메서드
        x = self.input.data
        gx = 2 * x * gy #도함수
        return gx

# exp function
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy #도함수
        return gx

# numerical differentitation
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

# 합성함수의 미분
def g(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

# 순전파
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)

# result: 3.297442541400256