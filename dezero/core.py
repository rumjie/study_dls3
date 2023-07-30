# step23 - dezero의 핵심이 되는 기능 복사해오기
import numpy as np
import contextlib
import weakref
import dezero

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndaray


class Variable:
    __array_priority__ = 200  # 연산자 우선순위

    def __init__(self, data, name=None):
        if data is not None:  # ndarray가 아닐 경우
            if not isinstance(data, array_types):  # step52
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.name = name  # 변수에 이름 붙이기
        self.grad = None  # 미분값 gradient 저장
        self.creator = None  # creator of variable
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func  # 메서드 추가
        self.generation = func.generation + 1  # 부모 함수의 세대보다 1 큰 수를 세대로 기록

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)  # core_simple.py 버전
            xp = dezero.cuda.get_array_modeul(self.data)  # step52
            self.grad = Variable(np.ones_like(self.data))

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

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)  # main backward
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

    def reshape(self, *shape):  # step38
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):  # step38
        return dezero.functions.transpose(self)

    @property  # 인스턴스 변수로 사용할 수 있게 해주는 데코레이터
    def T(self):
        return dezero.functions.transpose(self)

    def cleargrad(self):
        self.grad = None

    @property  # shape 메서드를 인스턴스 변수처럼 사용
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):  # 차원 수
        return self.data.ndim

    @property
    def size(self):  # 원소 수
        return self.data.size

    @property
    def dtype(self):  # 데이터 타입
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + "" * 9)
        return "variable(" + p + ")"

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()


# 순전파만 필요한 경우를 위한 모드
class Config:
    enable_backprop = True  # 역전파 활성 모드
    train = True


@contextlib.contextmanager  # 문맥을 판단하는 함수
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_mode():
    return using_config("train", False)  # test_mode 형태로 import 가능


# 순전파 코드만 실행할 때의 편의 함수
def no_grad():
    return using_config("enable_backprop", False)


# array로 변환
def as_array(x, array_module=np):  # 0차원의 nparray를 제곱하면 float이 되는 현상 방지
    if np.isscalar(x):
        return array_module.array(x)
    return x


# variable 로 전환
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)  # Variable이 아니라면 변환하여 리턴


# function class 구현
class Function:
    def __call__(self, *inputs):  # *표를 붙이면 임의 개수의 인수(가변 길이)를 넘겨 함수 호출 가능
        inputs = [as_variable(x) for x in inputs]

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
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # step40
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


# Mul class
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # step40
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0 * x1, gx1 * x0


# Neg class
class Neg(Function):  # function class 상속, 원하는 함수 클래스 구현
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


# Sub class
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # step40
        y = x0 - x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs  # step32
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gy, -gy


# Div class
class Div(Function):
    def forawrd(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = 0 / x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs  # step32
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


# Pow class
class Pow(Function):  # 밑이 x인 경우만 미분
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]  # step32
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


# function
def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


def neg(x):  # 파이썬 함수로 사용
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):  # 2-x 와 같은 연산을 수행하기 위함
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)  # 좌항과 우항의 순서를 바꾸면 결과가 달라지기 때문


def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div(x1, x0)  # 순서 바꿔 연산


def pow(x, c):
    return Pow(c)(x)


# 연산자 오버로드
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


class Parameter(Variable):  # step 44
    pass
