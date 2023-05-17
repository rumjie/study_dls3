import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.sin(x)
y.backward(create_graph=True)


for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)  # n차 미분

# results
# variable(-0.8414709848078965)
# variable(-0.5403023058681398)
# variable(0.8414709848078965)
