import os
import random
from PIL import Image
from matplotlib import pyplot
import RSC
import numpy as np

path = 'C:\\Projects\\OCR\\English\\A\\'
files = os.listdir(path)

for i in range(20):
	n = random.randint(0, len(files))
	data = np.asarray(Image.open(path+files[n]), dtype ='uint8')
	feat = RSC.generate_feature(data, 8)
	pyplot.plot(feat)

pyplot.show()

