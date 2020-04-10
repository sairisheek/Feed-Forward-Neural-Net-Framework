import pickle
import autodiff
import RSC
import numpy as np
from PIL import Image
import os
import random

featsize = 10
path = "C:\\Projects\\OCR\\English\\"
dirs = os.listdir(path)
wfile = open('weights.data', 'rb')
(w1,w2) = pickle.load(wfile)
bfile = open('bias.data', 'rb')
(b1,b2) = pickle.load(bfile)
z1 = autodiff.Entry("z1", "dot", ["w1","x", 'b1'], [True, False, True])
a1 = autodiff.Entry("a1", "tanh", ["z1"])
z2 = autodiff.Entry('z2', 'dot', ['w2','a1', 'b2'])
p = autodiff.Entry('f', 'softmax', ['z2'])
elist = [z1, a1, z2, p]
numimg = 100
vals = dict()
vals['w1'] = w1
vals['b1'] = b1
vals['w2'] = w2
vals['b2'] = b2
net = autodiff.WengertList(vals, elist)

hit = 0
size = 0
num = 0
avg = 0
for directory in dirs:
	if len(directory) == 1 and ord(directory) >= 65 and ord(directory) <= 90:
		files = os.listdir(path+directory)
		rng = np.random.default_rng()
		samples = 100 + rng.choice(len(files) - 100, size=100, replace=False)
		for i in samples:
			data = np.asarray(Image.open(path+directory+'\\'+files[i]), dtype ='uint8')
			feat = RSC.generate_feature(data, featsize)
			net.valMap['x'] = feat
			output = net.propagate()
			ind = np.argmax(output)
			if(ind == ord(directory)-65):
				hit += 1
			size += 1

		num += 1
		avg += hit/size
		print(hit/size)
		hit = 0
		size = 0

print('Average: ', avg/num)