import numpy as np
import math
import decimal
import matplotlib.pyplot as plt

T=np.arange(273,273+1105,5)
E=np.linspace(10,500, T.size)
zeta1=np.zeros(T.size**2)
zeta= zeta1.reshape(T.size,T.size)

for i in range (len(T)):
    for j in range (len(E)):
        zeta[i,j]=-((8.31*T[i]**2)/(5*E[j]*1000))*math.exp(-((E[j]*1000)/(8.31*T[i])))

mE=189.15
sigma=14.73
k0=1.1291*10**16
n=7.88

zeta2=(1-n)*k0*zeta

fE=np.zeros(len(E))

for i in range (len(E)):
    fE[i]=((1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((E[i]-mE)**2)/(2*sigma**2)))

W1=(1 + zeta2)**(1//1-n)
W=W1@fE
print(fE.sum())
plt.plot(T,W)
plt.show()
