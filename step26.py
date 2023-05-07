import numpy as np
from dezero import Variable

# from dezero.utils import get_dot_graph
from dezero.utils import _dot_var, _dot_func

# 시각화 코드 예시

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1  # random calculation

# # 변수 이름 지정
# x0.name = "x0"
# x1.name = "x1"
# y.name = "y"

# txt = get_dot_graph(y, verbose=False)
# print(txt)

# with open("Chapter3/step26_sample.dot", "w") as o:
#     o.write(txt)


x = Variable(np.random.randn(2, 3))
x.name = "x"
print(_dot_var(x))
print(_dot_var(x, verbose=True))


"""결과
4681765408 [label="x", color=orange, style=filled]
4681765408 [label="x: (2, 3) float64", color=orange, style=filled]
"""

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1
txt = _dot_func(y.creator)
print(txt)

"""결과
4681767520 [label="Add", color=lightblue, style=filled, shape=box]
4681766512 -> 4681767520
4681767376 -> 4681767520
4681767520 -> 4681767616
"""
