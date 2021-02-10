import numpy as np
import math
from scipy.optimize import minimize


def f(x):
	A=x[0]
	n=x[1]
	mE=x[2]
	sigma=x[3]
	T=473
	E = np.arange(10, 150, 1)
	fE = np.zeros(len(E))
	for i in range(len(E)):
		fE[i] = ((1/math.sqrt(2*math.pi*sigma**2)) *math.exp(-((E[i]-mE)**2)/(2*sigma**2)))

	t = np.array([0, 15, 30, 45, 60, 120, 180, 240, 300, 596, 1200, 1800])

	zeta1 = np.zeros(t.size*E.size)
	zeta = zeta1.reshape(t.size, E.size)

	for i in range(t.size):
	  for j in range(E.size):
	    zeta[i, j] = t[i]*math.exp((-(E[j]*1000)/(8.31*T)))

	zeta2 = (1-(1-n)*A*(10**8)*zeta)**(1/(1-n))
	final = zeta2@fE

	W1 = np.array([1, 0.775, 0.659, 0.6, 0.577, 0.472,0.418, 0.307, 0.177, 0.139, 0.082, 0])
	result1=(W1-final)**2
	obj=result1.sum()
	return obj

a=minimize(f,[30,5,70,15],method="Nelder-Mead")
print(a)
