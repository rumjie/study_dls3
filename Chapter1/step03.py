import numpy as np 

class Variable:
    def __init__(self, data):
        self.data=data

# function class 구현
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self,x):
        raise NotImplementedError()

# function 클래스를 상속, 입력값을 제곱하는 클래스
class Square(Function):
    def forward(self, x):
        return x ** 2

# exp function
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)