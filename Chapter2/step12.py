import numpy as np
import unittest


class SquareTest(unittest.TestCase):
    def test_forward(self):  # 제곱 함수 테스트하기
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        # result
        # Ran 1 test in 0.000s
        # OK

    def test_backward(self):  # 제곱 함수의 역전파 테스트하기
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        # result
        # Ran 2 tests in 0.000s
        # OK

    def test_gradient_check(self):  # 기울기 확인을 이용한 자동 테스트
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)  # 두 값이 가까운지 판정
        self.assertTrue(flg)


class Variable:
    def __init__(self, data):
        if data is not None:  # ndarray가 아닐 경우
            if not isinstance(data, np.ndarray):
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.grad = None  # 미분값 gradient 저장
        self.creator = None  # creator of variable

    def set_creator(self, func):
        self.creator = func  # 메서드 추가

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # backward 메서드 간소화 - y.grad 지정 생략 가능

        funcs = [self.creator]  # step07과 다른 부분 - 반복문으로 구현
        while funcs:
            f = funcs.pop()  # 함수를 가져옴
            x, y = f.input, f.output  # 함수의 입출력을 가져옴
            x.grad = f.backward(y.grad)  # backward 호출

            if x.creator is not None:
                funcs.append(x.creator)  # 리스트에 추가


# function class 구현
class Function:
    def __call__(self, *inputs):  # *표를 붙이면 임의 개수의 인수(가변 길이)를 넘겨 함수 호출 가능
        xs = [x.data for x in inputs]  # list comprehension
        ys = self.forward(*xs)  # unpack
        if not isinstance(ys, tuple):  # tuple이 아닌 경우
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # output 또한 리스트로 변경

        for output in outputs:
            output.set_creator(self)  # 출력 변수들에 창조자 설정
        self.inputs = inputs  # 입력 변수 보관
        self.outputs = outputs  # 출력 저장

        return outputs if len(outputs) > 1 else outputs[0]  # 리스트에 한 개만 들어있다면 그것만 리턴

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# Add class
class Add(Function):  # 변수를 직접 받아 변수로 돌려줌
    def forward(self, x0, x1):
        y = x0 + x1
        return y


# function 클래스를 상속, 입력값을 제곱하는 클래스
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):  # 역전파를 담당하는 메서드
        x = self.input.data
        gx = 2 * x * gy  # 도함수
        return gx


# exp function
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy  # 도함수
        return gx


# numerical differentitation
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# 합성함수의 미분
def g(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# 사용
# 파이썬 함수
def square(x):
    return Square()(x)  # ()(x)?


def exp(x):
    return Exp()(x)


def as_array(x):  # 0차원의 nparray를 제곱하면 float이 되는 현상 방지
    if np.isscalar(x):
        return np.array(x)
    return x


def add(x0, x1):
    return Add()(x0, x1)


# 개선된 add 활용
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)
# result: 5 , and 3 error

#### test ####
unittest.main()  # python Chapter1/step10.py 만 실행해서 테스트 수행하기
