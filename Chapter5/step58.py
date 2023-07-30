# 상위 폴더 참조
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dezero
import numpy as np
import dezero.functions as F
from PIL import Image
import dezero
from dezero.models import VGG16

url = "paste here"

img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]  # 배치용 축 추가

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file="vgg.pdf")  # graph visualization
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
