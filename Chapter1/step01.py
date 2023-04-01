# Variable class
class Variable:
    def __init__(self, data):
        self.data=data

# print data
import numpy as np
data=np.array(1.0)
x=Variable(data)
print(x.data)
#python Chapter1/step01.py -> 1.0

x.data=np.array(2.0) #수정 가능
print(x.data)

# x는 데이터를 담은 상자
# result:
# 1.0
# 2.0