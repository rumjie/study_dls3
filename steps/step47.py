# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(
    np.array(
        [
            [
                1,
                2,
                3,
            ],
            [4, 5, 6],
        ]
    )
)
y = F.get_item(x, 1)  # temporary
y.backward()
