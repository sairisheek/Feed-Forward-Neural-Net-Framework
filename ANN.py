import numpy as np
import RSC
import autodiff
import os
from PIL import Image
import pickle
import random

path = "C:\\Projects\\OCR\\English\\"
dirs = os.listdir(path)

def train(net, x, y, alpha):
	loss = 0
	h1 = 100
	(w1delta, b1delta, w2delta, b2delta) = (0, 0, 0, 0)
	for i in range(len(x)):
		net.valMap["x"] = x[i]
		net.valMap["y"] = y[i]
		loss += net.propagate()
		net.backpropagate()
		#loss = net.gradient_check('w1', 1e-10)
		w1delta += net.delta['w1']
		b1delta += net.delta['b1']
		w2delta += net.delta['w2']
		b2delta += net.delta['b2']
		net.reset()
	net.valMap['w1'] -= alpha*w1delta
	net.valMap['b1'] -= alpha*b1delta
	net.valMap['w2'] -= alpha*w2delta
	net.valMap['b2'] -= alpha*b2delta
	#print(np.linalg.norm(net.delta['w']))
	return loss

featsize = 10
h1 = 100


w1 = np.random.normal(0,np.sqrt(2/(h1 + 2*featsize)),(h1,2*featsize))
b1 = np.zeros(h1)
z1 = autodiff.Entry("z1", "dot", ["w1","x", 'b1'], [True, False, True])
a1 = autodiff.Entry("a1", "tanh", ["z1"])
w2 = np.random.normal(0, np.sqrt(2/(26 + h1)), (26, h1))
b2 = np.zeros(26)
z2 = autodiff.Entry('z2', 'dot', ['w2','a1', 'b2'])
p = autodiff.Entry('p', 'softmax', ['z2'])
f = autodiff.Entry("f", "crossentropy", ["y","p"], [False,True])

vals = dict()
vals["w1"] = w1
vals['b1'] = b1
vals['w2'] = w2
vals['b2'] = b2
elist = [z1,a1,z2,p,f]
net = autodiff.WengertList(vals, elist)
ep = 1
numimg = 100

featlist = np.zeros((26*numimg, 2*featsize))
labellist = np.zeros((26*numimg, 26))

fi = 0
li = 0
for directory in dirs:
	if len(directory) == 1 and ord(directory) >= 65 and ord(directory) <= 90:
		files = os.listdir(path+directory)
		for i in range(numimg):
			data = np.asarray(Image.open(path+directory+'\\'+files[i]), dtype ='uint8')
			feat = RSC.generate_feature(data, featsize)
			label = np.zeros(26)
			label[ord(directory)-65] = 1
			featlist[fi] = feat
			labellist[li] = label
			li += 1
			fi += 1

l = 900                                                                     
while(l > 130):
	l = train(net, featlist, labellist, .001)
	print(l)
	#print(net.valMap)
	#print(net.delta)
wfile = open('weights.data', 'wb')
pickle.dump((net.valMap['w1'], net.valMap['w2']), wfile)
bfile = open('bias.data', 'wb')
pickle.dump((net.valMap['b1'],net.valMap['b2']), bfile)
