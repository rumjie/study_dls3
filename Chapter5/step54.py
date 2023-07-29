# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dezero
import dezero.functions as F
import numpy as np

x = np.ones(5)
print(x)

# 학습
y = F.dropout(x)
print(y)

# 테스트
with test_mode():
    y = F.dropout(x)
    print(y)
