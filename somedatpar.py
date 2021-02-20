import numpy as np
from scipy import optimize
import scipy as sp
import pandas as pd 
import math
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

def multistart(fun, N, full_output = False, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    res_list = np.empty(N, dtype = object)
    def f(i):    
        x0=np.zeros(4)
        for j in range (4):
            x0[j]=np.random.uniform(1,100)
        res = sp.optimize.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        return res
    num_cores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in range(N))
    res_list=np.array(result)
    sort_res_list=res_list[np.argsort([res.fun for res in res_list])]
    if full_output:
        return sort_res_list[0], sort_res_list
    else:
        return sort_res_list[0]
        
a=pd.read_excel('AP5kpm.xlsx',usecols='A,C')
dataset=a.to_numpy()
T=dataset[:,0]
Wi=dataset[:,1]/100
W=(Wi-Wi[Wi.size-1])/(1-Wi[Wi.size-1])

E=np.arange(10,500,1)
zeta1=np.zeros(T.size*E.size)
zeta=zeta1.reshape(T.size,E.size)
for i in range(T.size):
    for j in range(E.size):
        zeta[i,j]=-((8.31*T[i]**2)/(5*E[j]*1000))*math.exp(-(E[j]*1000)/(8.31*T[i]))

def objective_function(x):
    n=x[0] 
    k0=x[1]
    mE =x[2]
    sigma=x[3]
    fE=np.zeros(len(E))
    for i in range(len(E)):
        fE[i] = ((1/math.sqrt(2*math.pi*sigma**2)) *math.exp(-((E[i]-mE)**2)/(2*sigma**2)))
    est=(1+(1-n)*k0*(10**15)*zeta)**(1/(1-n))
    W1=est@fE
    obj1=(W-W1)**2
    fitness=obj1.sum()
    return fitness

sp.random.seed(0)
res, res_list_2 = multistart(objective_function,300,full_output=True)

x=res.x
n=x[0] 
k0=x[1]
mE =x[2]
sigma=x[3]
fE=np.zeros(len(E))
for i in range(len(E)):
    fE[i] = ((1/math.sqrt(2*math.pi*sigma**2)) *math.exp(-((E[i]-mE)**2)/(2*sigma**2)))
est=(1+(1-n)*k0*(10**15)*zeta)**(1/(1-n))
W1=est@fE

plt.plot(T,W,T,W1)
plt.show()