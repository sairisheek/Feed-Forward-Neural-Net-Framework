from PIL import Image
from matplotlib import pyplot
import numpy as np 


def center_of_mass(data):
	xcum = 0
	ycum = 0
	data = 1 - (data/255)
	sum = np.sum(data)
	(x,y) = np.meshgrid(np.arange(data.shape[0]),np.arange(data.shape[1]))
	xcum = np.sum(np.multiply(x,data))
	ycum = np.sum(np.multiply(y,data))	

	return (round(xcum/sum), round(ycum/sum))

def find_radius(data, com):
	(x,y) = np.meshgrid(np.arange(data.shape[0]) - com[0],np.arange(data.shape[1]) - com[1])
	euclidean_dist = np.sqrt(np.square(x) + np.square(y))
	radius = 0
	euclidean_dist = np.multiply((1 - data/255), euclidean_dist)
	return euclidean_dist

def get_cutpoints(data, com, radius, n, euclidean_dist):
	angles = np.arange(2*n)*np.pi/n;
	endpoints = np.array([com[0] + radius*np.cos(angles), com[1] + radius*np.sin(angles)]).T
	maxes = np.zeros(2*n) 
	means = np.zeros(2*n)

	for k in  range(2*n):
		num = max(np.abs(endpoints[k][0] - com[0]), np.abs(endpoints[k][1] - com[1]))
		x, y = np.linspace(com[0], endpoints[k][0], int(np.ceil(num))), np.linspace(com[1], endpoints[k][1], int(np.ceil(num)))
		x, y = x[(x >= 0) & (x < data.shape[1])], y[(y >= 0) & (y < data.shape[1])]
		if x.size > y.size:
			x = x[:y.size]
		else:
			y = y[:x.size]

		x = x.astype(np.int)
		y = y.astype(np.int)
		
		numpoints = 0
		for i in range(x.size - 1):
			prev = (y[i], x[i])
			nex = (y[i+1], x[i+1]) 
			if(euclidean_dist[prev] > 0 and euclidean_dist[nex] == 0):
				#pyplot.plot(prev[1], prev[0], 'ro')
				maxes[k] = max(euclidean_dist[prev], maxes[k])
				means[k] += euclidean_dist[prev]
				numpoints += 1
			elif (euclidean_dist[nex] > 0 and euclidean_dist[prev] == 0):
				#pyplot.plot(nex[1], nex[0], 'ro')
				maxes[k] = max(euclidean_dist[nex], maxes[k])
				means[k] += euclidean_dist[nex]
				numpoints += 1
		if(numpoints == 0):
			maxes[k] = radius
			means[k] = radius
		else:
			means[k] /= numpoints

	return (maxes, means)

def axis_of_reference(maxes):
	n = int(len(maxes)/2)
	minloss = float('inf')
	minaxis = 0
	for i in range(n):
		loss = 0
		for o in range(1,n):
			loss += np.abs(maxes[i+o] - maxes[i-o])
		if loss < minloss:
			minloss = loss
			minaxis = i
		#print(i,' ',loss)
	if(maxes[minaxis] < maxes[minaxis + n]):
		minaxis = minaxis + n
	return minaxis

def generate_feature(data, n):
	data = np.where(data <  20, 0,255)
	com = center_of_mass(data)
	dist = find_radius(data, com)
	rad = np.max(dist)
	(maxes, feat) = get_cutpoints(data, com, rad+1, n, dist)
	axis = axis_of_reference(maxes)
	feat = np.roll(feat,-1*axis)/rad
	#feat = np.append(feat, 1)
	return feat
'''
data = np.asarray(Image.open('P.png'), dtype ='uint8')
data = np.where(data <  20, 0,255)
n = 8
com = center_of_mass(data)
print(com)
dist = find_radius(data, com)
rad = np.max(dist)
print(rad)
(maxes, feat) = get_cutpoints(data, com, rad+1, n, dist)
axis = axis_of_reference(maxes)
print(axis)
feat = np.roll(-1*axis)
(ax, ay) = (com[0] + rad*np.cos(axis*np.pi/n), com[1] + rad*np.sin(axis*np.pi/n))


pyplot.imshow(data, cmap='gray')
pyplot.plot(com[0], com[1], 'bo')


pyplot.plot([com[0], ax], [com[1], ay])
pyplot.show()
'''



