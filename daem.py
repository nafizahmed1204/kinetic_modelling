import numpy as np
import math
import matplotlib.pyplot as plt

def W(T,A,n,mE,sigma):
	E=np.arange(10,150, 1)
	fE=np.zeros(len(E))
	for i in range (len(E)):
		fE[i]=((1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((E[i]-mE)**2)/(2*sigma**2)))  

	t=np.array([0, 15, 30, 45, 60, 120, 180, 240, 300, 596, 1200, 1800])
	
	zeta1=np.zeros(t.size*E.size)
	zeta=zeta1.reshape(t.size,E.size) 
	
	for i in range (t.size):
		for j in range (E.size):
			zeta[i,j]=t[i]*math.exp((-(E[j]*1000)/(8.31*T)))
	
	zeta2=(1-(1-n)*A*(10**8)*zeta)**(1/(1-n))
	final=zeta2@fE 
	return final,fE,t

b1,b2,b3=W(473,34.7854,1.00000575,105.65175085,6.18131)

W1=np.array([1, 0.775, 0.659, 0.6, 0.577, 0.472, 0.418, 0.307, 0.177, 0.139, 0.082, 0])

obj=(W1-b1)**2
result1=obj.sum()
print(result1)

plt.plot(b1,b3,W1,b3)
plt.show()



