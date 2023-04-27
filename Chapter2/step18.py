import numpy as np
import unittest
import weakref
import contextlib


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
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func  # 메서드 추가
        self.generation = func.generation + 1  # 부모 함수의 세대보다 1 큰 수를 세대로 기록

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # backward 메서드 간소화 - y.grad 지정 생략 가능

        funcs = []  # list
        seen_set = set()

        def add_funcs(f):  # backward 메서드의 중첩 함수
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)  # 집합을 이용해 중복 추가 방지
                funcs.sort(key=lambda x: x.generation)  # 함수를 세대 순으로 정렬

        add_funcs(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 가져옴
            gys = [output().grad for output in f.outputs]  # weakref for outputs
            gxs = f.backward(*gys)  # 역전파 호출
            if not isinstance(gxs, tuple):  # 튜플이 아니라면 변환
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # 미분값을 grad에 저장
                if x.grad is None:
                    x.grad = gx  # 미분값 첫 설정
                else:
                    x.grad = x.grad + gx  # 전달된 미분값을 더함

                if x.creator is not None:
                    add_funcs(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한참조

    def cleargrad(self):
        self.grad = None


# function class 구현
class Function:
    def __call__(self, *inputs):  # *표를 붙이면 임의 개수의 인수(가변 길이)를 넘겨 함수 호출 가능
        xs = [x.data for x in inputs]  # list comprehension
        ys = self.forward(*xs)  # unpack
        if not isinstance(ys, tuple):  # tuple이 아닌 경우
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # output 또한 리스트로 변경

        if Config.enable_backprop:  # True 일때만 역전파 코드 실행
            self.generation = max([x.generation for x in inputs])  # setting generation

            for output in outputs:
                output.set_creator(self)  # 출력 변수들에 창조자 설정
            self.inputs = inputs  # 입력 변수 보관, 참조 카운트 증가
            self.outputs = [
                weakref.ref(output) for output in outputs
            ]  # self.outputs가 대상을 약한 참조로 가리키게 변경

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

    def backward(self, gy):
        return gy, gy


# function 클래스를 상속, 입력값을 제곱하는 클래스
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):  # 역전파를 담당하는 메서드
        x = self.inputs[0].data
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


# 순전파만 필요한 경우를 위한 모드
class Config:
    enable_backprop = True  # 역전파 활성 모드


@contextlib.contextmanager  # 문맥을 판단하는 함수
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


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


# 순전파 코드만 실행할 때의 편의 함수
def no_grad():
    return using_config("enable_backprop", False)


# 중간 미분값 제거
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad)  # None
print(x0.grad)  # 2.0

# 메모리 사용량 확인
import psutil

p = psutil.Process()
rss = p.memory_info().rss / 2**20  # Bytes to MB

Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()
print(f"memory usage: {rss: 10.5f} MB")

# Config.enable_backprop = False
# x = Variable(np.ones((100, 100, 100)))
# print(type(square(x)))
# y = square(square(square(x)))  # square(x) 값이 nonetype
# print(f"memory usage: {rss: 10.5f} MB")

# with 문으로 역전파 비활성 모드 전환
with using_config("enable_backprop", False):  # == with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

#### test ####
unittest.main()  # python Chapter1/step10.py 만 실행해서 테스트 수행하기
