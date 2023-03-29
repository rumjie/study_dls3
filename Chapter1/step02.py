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

x = Variable(np.array(3))
f = Square()
y = f(x)
print(type(y))
print(y.data)      