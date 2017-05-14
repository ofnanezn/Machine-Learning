import numpy as np
import matplotlib.pyplot as plt
import random

""" Load training data from file.
	Consists of tuples (x_i,y_i) where x_i is the indepent variable
	and y_i is the dependent variable. """

def load_training_data(filename):
	f = open(filename, 'r')
	training_data = np.empty((0,2),float)
	for line in f:
		x, y = map(float, line.strip().split(" "))
		training_data = np.vstack((training_data, [x,y]))
	return training_data

""" Analytic method:
	w1 and w2 are the result of derivating partially the SME (Loss(h))
	of h (hypothesis), then resolve equations for w1 and w2.
	Finally just replace in model y = w1*x + w0 """

def w1_opt(N,X,Y):
	sum_xj_yj = sum(X[j]*Y[j] for j in range(len(X)))
	sum_x = sum(X)
	sum_xj2 = sum([X[j]**2 for j in range(len(X))])
	return (N * sum_xj_yj - sum_x * sum(Y)) / (N * sum_xj2 - sum_x**2)

def w0_opt(N,X,Y,w1):
	return (sum(Y) - w1 * sum(X))/N

""" Gradient Descent method: """

def hypothesis(w0,w1,X):
	return w1*X + w0

def loss_func(Y,H):
	#print(H)
	return sum((Y-H)**2)


def gradient_descent(X,Y,alpha,epsilon):
	w0 = random.uniform(-10,10)
	w1 = random.uniform(-10,10)
	h_x = hypothesis(w0,w1,X) 
	loss = delta = loss_func(Y,h_x)
	while delta > epsilon:
		sum_y_h = sum(Y-h_x)
		sum_y_h_x = sum([(Y[j]-h_x[j])*X[j] for j in range(len(X))])
		w0 = w0 + alpha * sum_y_h
		w1 = w1 + alpha * sum_y_h_x 
		#print(w0,w1)
		h_x = hypothesis(w0,w1,X)
		loss2 = loss_func(Y,h_x)
		delta = abs(loss-loss2)
		print(delta)
		loss = loss2 
	return w0, w1

""" Results of the model """

def linear_model(w0,w1,x):
	return w1*x + w0

def print_model(w0,w1):
	print("y = "+str(w1)+"x + "+str(w0) if w0 > 0 else "y = "+str(w1)+"x "+str(w0))

def main():
	training_data = load_training_data('data01.txt')
	X = training_data[:,0]
	Y = training_data[:,1]
	#Analytic Method
	w1 = w1_opt(len(training_data),X,Y)
	w0 = w0_opt(len(training_data),X,Y,w1)
	print("Model obtained with Analytic Method:")
	print_model(w0,w1)
	#Gradient Descent Method
	w0,w1 = gradient_descent(X,Y,0.1,0.000001)
	print("Model obtained with Gradient Descent:")
	print_model(w0,w1)
	#Plot model
	for i in range(len(X)):
		plt.plot(X[i],Y[i],'o')
		plt.hold(True)
	t = np.linspace(0,1,400)
	plt.plot(t,linear_model(w0,w1,t))
	plt.show()

if __name__ == "__main__":
    main()
