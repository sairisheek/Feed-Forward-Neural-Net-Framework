import numpy as np
import operator

class WengertList:
	
	def __init__(self, val, entrylist):
		self.entryList = entrylist
		self.valMap = val
		self.delta = dict()

	def propagate(self):
		for entry in self.entryList:
			op = entry.fop
			self.valMap[entry.name] = op(*list(map(self.valMap.get, entry.vars)))
		return self.valMap["f"]

	def backpropagate(self):
		if type(self.valMap["f"]) is np.ndarray:
			self.delta["f"] = np.ones(self.valMap["f"].shape)
		else:
			self.delta["f"] = 1
		for entry in reversed(self.entryList):
			for i in range(len(entry.aop)):
				op = entry.aop[i]
				if entry.computeDelta[i]:
					if entry.vars[i] not in self.delta:
						self.delta[entry.vars[i]] = 0
					dop = entry.dop
					#print(entry.name)
					#print(entry.vars[i])
					#print(self.valMap)
					self.delta[entry.vars[i]] += dop(self.delta[entry.name],op(*list(map(self.valMap.get, entry.vars))))
	def gradient_check(self, parameter, epsilon):
		loss  = self.propagate()
		self.backpropagate()
		theta = self.valMap[parameter]
		grad_approx = np.zeros(theta.shape) 
		grad = self.delta[parameter]

		for i in range(theta.shape[0]):
			for j in range(theta.shape[1]):
				eps = np.zeros(theta.shape)
				eps[i][j] = epsilon
				theta_plus = theta + eps
				theta_minus = theta - eps
				self.valMap[parameter] = theta_plus
				loss_plus = self.propagate()
				self.valMap[parameter] = theta_minus
				loss_minus = self.propagate()
				grad_approx[i][j] = (loss_plus - loss_minus)/(2*epsilon)
		
		diff = np.linalg.norm(grad - grad_approx)/(np.linalg.norm(grad) + np.linalg.norm(grad_approx))
		print(diff)
		return loss



	def reset(self):
		for k in self.delta:
			self.delta[k] = 0

class Entry():
	def asigmoid(a):
		sig = 1/(1 + np.exp(-1*a))
		return sig*(1-sig)
	def softmax(a):
		exps = np.exp(a - np.max(a))
		return exps / np.sum(exps)
	def asoftmax(a):
		''''
		p = np.exp(a-np.max(a))
		p /= np.sum(p)
		p *= (1 - np.sum(p))
		'''
		a = Entry.softmax(a)
		a = np.diag(a) -np.outer(a,a)
		return a
	def specialdot(a,b):
		if not hasattr(b, 'shape'):
			return a
		elif(len(b.shape) == 2):
			return np.dot(b.T, a)
		elif len(b.shape) == 1:
			return np.outer(a,b)
	def fregularize(a,b):
		return a + np.sum(np.square((b)))


	functionMap = {
		"add": lambda a,b: a+b,
		"square": lambda a:a*a,
		"multiply": lambda a,b: a*b,
		"dot": lambda a,b,c: np.dot(a,b) + c,
		"sigmoid": lambda a: 1/(1 + np.exp(-1*a)),	
		"tanh": np.tanh,
		"softmax": softmax,
		"crossentropy": lambda a,b: np.sum(-1*np.multiply(a,np.log(b+1e-10))),
		"regularize": fregularize
		}

	adjointMap = {
		"add": [lambda a,b: 1, lambda a,b: 1],
		"square": [lambda a:2*a],
		"multiply": [lambda a,b: b, lambda a,b: a],
		"dot": [lambda a,b,c: b, lambda a,b,c:a, lambda a,b,c:7],
		"sigmoid": [asigmoid],
		"tanh": [lambda a: 1 - np.square(np.tanh(a))],
		"softmax":[asoftmax],
		"crossentropy":[lambda a,b: a,lambda a,b: (-1*np.divide(a,b)).T],
		"regularize": [lambda a,b: 0, lambda a,b: 2*b]
	}

	deltaOpMap = {
		"add":operator.mul,
		"square":operator.mul,
		"multiply":operator.mul,
		"dot":specialdot,
		"sigmoid":np.multiply,
		"tanh":np.multiply,
		"softmax":np.dot,
		"crossentropy":np.dot,
		"regularize":np.add
	}

	def __init__(self, name, op, vars, computeDelta=None):
		self.name = name
		self.fop = self.functionMap[op]
		self.aop = self.adjointMap[op]
		self.dop = self.deltaOpMap[op]
		self.vars = vars
		if computeDelta == None:
			self.computeDelta = [True]*len(vars)
		else:
			self.computeDelta = computeDelta
		
		
'''
v1 = Entry("v1", "multiply", ["x1", "x2"])
v2 = Entry("v2", "square", ["x2"])
v3 = Entry("v3", "add", ["v1", "v2"])
f = Entry("f", "add", ["x3", "v3"], [False, True])

elist = [v1,v2,v3,f]
val = {"x1": 2, "x2" : 4, "x3":9}
wl = WengertList(val, elist)
wl.propagate()
wl.backpropagate()
print(wl.delta)
print(wl.valMap)

f = Entry("f", "sigmoid", ["r1"])
r1 = Entry("r1", "dot", ["w1", "x1"], [True, False])
val2= {"w1":np.asarray([[1,2,3],
						[9,8,7],
						[4,5,6]]), 
		"x1": np.asarray([[1,2,3]]).T}

wl2 = WengertList(val2,[r1,f])
wl2.propagate()
print(wl2.valMap)
wl2.backpropagate()

print(wl2.delta)
'''