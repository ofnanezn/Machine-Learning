import numpy as np
import matplotlib.pyplot as plt

def load_training_data(filename):
	f = open(filename, 'r')
	training_data = np.empty((0,2),float)
	for line in f:
		x, y = map(float, line.strip().split(" "))
		training_data = np.vstack((training_data, [x,y]))
	return training_data

def w1_opt(N,X,Y):
	sum_xj_yj = sum(X[j]*Y[j] for j in range(len(X)))
	sum_x = sum(X)
	sum_xj2 = sum([X[j]**2 for j in range(len(X))])
	return (N * sum_xj_yj - sum_x * sum(Y)) / (N * sum_xj2 - sum_x**2)

def w0_opt(N,X,Y,w1):
	return (sum(Y) - w1 * sum(X))/N

def linear_model(w0,w1,x):
	return w1*x + w0

def print_model(w0,w1):
	print("y = "+str(w1)+"x + "+str(w0) if w0 > 0 else "y = "+str(w1)+"x "+str(w0))   

def main():
	training_data = load_training_data('data01.txt')
	X = training_data[:,0]
	Y = training_data[:,1]
	w1 = w1_opt(len(training_data),X,Y)
	w0 = w0_opt(len(training_data),X,Y,w1)
	print("Model obtained:")
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
