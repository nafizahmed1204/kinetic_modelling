import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#density profile determination
t_step=0.1*10**(-3) #s
k0=1.1*10**4
E=59.5 #kJ/mol
Rg=8.31 #J/molK
n=5000
m=100
rho=np.zeros(n)
t=np.zeros(n)
rho[0]=440 #kg/m3
rhoc=np.zeros(n)
rhoc[0]=0
t[0]=0
a=0
vc=0.084
rhob=np.zeros(n)
rhob[0]=440
cpb=1500+1273
cpc=420+2.09*1273+(6.85*10**(-4))*1273**2
cp=np.zeros(n)
l=np.zeros(n)
h=np.zeros(n)

#for velocity calculation
ug=0.011 #m/s
Dp=550*10**(-6)
r=Dp/2
space_step=r/100
miu_g=41.52*10**(-6)
rho_g=1.16
Rep=ug*Dp*rho_g/miu_g
CD=(24/Rep)*(1+0.15*Rep**0.687)
up=np.zeros(len(t))
up[0]=2.12*10**(-3)
up_settling=0

#temperature profile first part
Red=ug*50*(10**(-3))*rho_g/miu_g
Pr=0.76
Ts=np.zeros(n)
Ts[0]=300
#for solving linear equation


for i in range (n+1):
    if i+2==n+1:
        break
    else: 
        rhob[i+1]=rhob[i]-k0*t_step*rhob[i]*np.exp(-((E*1000)/(Rg*1273)))
        rhoc[i+1]=((rhob[i]-rhob[i+1])*vc)+rhoc[i]
        rho[i+1]=(rhob[i+1])+(rhoc[i+1])
        t[i+1]=t[i]+t_step
        up[i+1]=up[i]+t_step*(((3*CD*rho_g*abs(ug-up[i])*(ug-up[i]))/(4*Dp))+(rho[i]-rho_g)*9.81/rho[i])
        if up[i+1]-up[i]<0.0001:
            up_settling=up[i+1]

for i in range(n):
    cp[i]=(rhob[i]*cpb+rhoc[i]*cpc)/rho[i]
    l[i]=(rhob[i]*0.1256+rhoc[i]*0.0837)/rho[i]
    h[i]=(l[i]/(50*10**(-3)))*(2+0.6*(Red**0.5)*Pr**(1/3))
   
for i in range(n): #getting surface temperatures in my own way. 
    if i+1<n:
        Ts[i+1]=Ts[i]+(t_step*(h[i]*(1273-Ts[i])+5.67*(10**(-8))*0.95*(1273**4-Ts[i]**4)))/(cp[i]*rho[i])

Tp=np.full((n,m),300)
for i in range(n): #inputting the surface boundary condition 
    Tp[i,m-1]=Ts[i]

A=np.zeros((m-2,m-2))
rhm=np.zeros(m-2) #right hand sides of the linear equations 
x=np.zeros((n-1,m-2))

r_step=r/m
re=r_step/2
rw=re+r_step*2
r_matrix=np.zeros(m-2)
r_matrix[0]=re

ae=np.zeros((n-1,m-2))
aw=np.zeros((n-1,m-2))
ap=np.zeros((n-1,m-2))
ap0=np.zeros((n-1,m-2))
b=np.zeros((n-1,m-2))

q_pyro=418000
for i in range(n-1):
    for j in range(m-2):
        ae[i,j]=re*l[i+1]*0.5/r_step
        aw[i,j]=rw*l[i+1]*0.5/r_step
        ap[i,j]=ae[i,j]+aw[i,j]+(rho[i+1]*cp[i+1]*(abs(re**2-rw**2))/(2*t_step))
        ap0[i,j]-ap[i,j]+2*(rho[i+1]*cp[i+1]*(abs(re**2-rw**2))/(2*t_step))
        b[i,j]=ae[i,j]*Tp[i,j]+aw[i,j]*Tp[i,j+1]+(q_pyro)*((abs(re**2-rw**2))/(2*t_step))*(rho[i]-rho[i+1])
        A[j,j]=ap[i,j]
        if j+1<m-2:
            A[j+1,j]=-ae[i,j]
            A[j,j+1]=-aw[i,j]
        re=re+r_step
        rw=rw+r_step
        if j+1<m-2:
            r_matrix[j+1]=re

    rhm[0]=b[i,0]+ap0[i,0]*Tp[i,1]
    rhm[m-3]=b[i,m-3]+ap0[i,m-3]*Tp[i,m-2]+(r*l[i]/r_step)*0.5*Tp[i+1,m-1]
    for k in range(m-3):
        if k+1<m-3:
            rhm[k+1]=b[i,k+1]+ap0[i,k+1]*Tp[i,k+2]
    result=np.linalg.solve(A,rhm)
    x[i]=result

print(x)
rad,time=np.meshgrid(r_matrix,t[1:])
fig=plt.figure(figsize=[8,8])
ax=fig.gca(projection='3d')
ax.plot_surface(rad,time,x)
plt.show()