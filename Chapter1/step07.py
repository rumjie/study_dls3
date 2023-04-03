import numpy as np 

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 미분값 gradient 저장
        self.creator = None # creator of variable
    
    def set_creator(self, func):
        self.creator = func # 메서드 추가

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() # 재귀적으로 앞 변수의 backward 호출

# function class 구현
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 출력 변수에 창조자 설정
        self.input = input # 입력 변수 보관
        self.output = output # 출력 저장
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

# 거꾸로 거슬러 올라간다
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x
# assert: 결과가 True 가 아니면 예외 발생

y.grad = np.array(1.0)

C = y.creator # 함수 가져오기
b = C.input # 함수의 input 가져오기
b.grad = C.backward(y.grad) # backward 가져오기

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)

print(b.grad, a.grad, x.grad)
# result : 2.568050833375483 3.297442541400256 3.297442541400256

# 함수를 사용해 결과 출력하기
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

