
"""
Created on Tue Feb 28 19:19:00 2017

@author: hammed fatoyinbo
"""
import scipy
import numpy as np
from scipy import integrate
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
#% matplotlib inline


#-Define parameter
v1=-0.027
v2=0.025
v4=0.0145
v5=0.008
v6=-0.015
ca3=4.0e-7
ca4=1.5e-7
phi=2.664
vl=-0.07
vk=-0.09
vca=0.08
gk=3.1416e-13
gl=7.854e-14
gc=1.57e-13
c=1.9635e-14
kd= 1.0e-6
bt=1.0e-4
alpha=7.9976e7
kca=1.3567537e2



# define system in terms of separated differential equations
#-useful functions
def f(n,v,ca):
    v3=(-v5/2)*np.tanh((ca-ca3)/ca4)+v6
    return  phi*np.cosh((v-v3)/(2*v4))*(0.5*(1+np.tanh((v-v3)/v4))-n)
    
def g(n,v,ca):
    return 1/c*(-gl*(v-vl)-gk*n*(v-vk)-gc*0.5*(1+np.tanh((v-v1)/v2))*(v-vca))

def h(n,v,ca):
    return ((-alpha*gc*0.5*(1+np.tanh((v-v1)/v2))*(v-vca)-kca*ca)*((kd+ca)**2/((kd+ca)**2+kd*bt)))


# initialize lists containing values

n  = []
v  = []
ca = []
    

    #-Main program
def sys(n_1, v_1, ca_1, dt, t):
   
    n.append(n_1)
    v.append(v_1)
    ca.append(ca_1)
        
    for i in range(t):
        n.append(n[i]+(f(n[i],v[i],ca[i]))*dt)
        v.append(v[i]+(g(n[i],v[i],ca[i]))*dt)
        ca.append(ca[i]+(h(n[i],v[i],ca[i]))*dt)
    return n, v, ca


#-Call the main program
#n_output,v_output,ca_output = 

sys(0, 0 , 0, 0.01, 1000)

'''
#plot
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)

#-Plot the output
ax1 = fig.add_subplot(221)
ax1.plot(n, 'g-',label='n(t)')
ax1.set_title("Dynamics in time")
ax1.set_xlabel("time")
ax1.legend(loc='best')

ax3 = fig.add_subplot(223)
ax3.plot(v, 'r-', label='v(t)')
ax3.set_title("Dynamics in time")
ax3.set_xlabel("time")
ax3.legend(loc='best')

ax2 = fig.add_subplot(222)
ax2.plot(ca, 'b-', label='ca(t)')
ax2.set_title("Dynamics in time")
ax2.set_xlabel("time")
ax2.legend(loc='best')



ax4 = fig.add_subplot(224)
ax4.plot( n, v, color="blue")
ax4.set_xlabel("n")
ax4.set_ylabel("v")  
ax4.set_title("Phase space")
ax4.grid()

#fig = plt.figure(figsize=(15,5))

#ax1 = fig.add_subplot(221,projection='3d')
#ax1.plot( n, v,ca, color="blue")
#ax1.set_xlabel("n(t)")
#ax1.set_ylabel("v(t)")  
#ax1.set_zlabel("ca(t)")
#ax1.set_title("3D plot")
#ax1.grid()

'''

####################################Function for Jacobian#####################################################
def Jacob(n,v,ca):
    k  = 1E-5
    fn =  (f(n+k,v,ca)-f(n,v,ca))*(k**(-1.0))
    fv =  (f(n,v+k,ca)-f(n,v,ca))*(k**(-1.0))
    fca = (f(n,v,ca+k)-f(n,v,ca))*(k**(-1.0))
    gn =  (g(n+k,v,ca)-g(n,v,ca))*(k**(-1.0))
    gv =  (g(n,v+k,ca)-g(n,v,ca))*(k**(-1.0))
    gca = (g(n,v,ca+k)-g(n,v,ca))*(k**(-1.0))
    hn =  (h(n+k,v,ca)-h(n,v,ca))*(k**(-1.0))
    hv =  (h(n,v+k,ca)-h(n,v,ca))*(k**(-1.0))
    hca =  (h(n,v,ca+k)-h(n,v,ca))*(k**(-1.0))   
    return [[fn,fv,fca],[gn,gv,gca],[hn,hv,hca]]

############ for the  equilibrium points###############################################


def steady(x, t, p):
    """
    To calculate the equilibria

    Arguments:
        w :  vector of the state variables:
                  w = [n, v, ca]
        t :  time
        p :  vector of the parameters:
                  p = [v1,v2,v4,v5,v6,ca3,ca4,phi,vl,vk,vca,gk,gl,gca,c,kd,bt,alpha,kca]
    """

    n, v, ca = x
    v1,v2,v4,v5,v6,ca3,ca4,phi,vl,vk,vca,gk,gl,gca,c,kd,bt,alpha,kca = p
    
   
    v3=(-v5/2)*np.tanh((ca-ca3)/ca4)+v6

    f = [ phi*np.cosh((v-v3)/(2*v4))*(0.5*(1+np.tanh((v-v3)/v4))-n),
         1/c*(-gl*(v-vl)-gk*n*(v-vk)-gca*0.5*(1+np.tanh((v-v1)/v2))*(v-vca)),
        ((-alpha*gca*0.5*(1+np.tanh((v-v1)/v2))*(v-vca)-kca*ca)*((kd+ca)**2/((kd+ca)**2+kd*bt))) ]
    return f

# the  parameter values
V_1 = np.linspace(-0.0296,-0.0196, 100)
#V_1 = np.linspace(-0.0181,-0.0296, 100)
F=np.zeros((0,3))
P=np.zeros((0,3))
Q=np.zeros((0,3))
R=np.zeros((0,3))
#x0 = [9.11458e-3, -5.74219e-2, -4.55729e-09]
x0 = [2.50243680e-01, -1.89735122e-02,  -5.71248696e-08]
x1 = [2.73592e-01, -1.83627e-02, -5.74296e-08]
x2 = [2.8838e-01, -1.78697e-02, -6.0757e-08]
x3 = [2.98239e-01, -1.7501e-02, -6.32218e-08]
eq=x0
eq_1= x1
eq_2= x2
eq_2= x3
K=[]
if __name__ == "__main__":
  
    for v1 in V_1:
    #v1=-0.0249476
    	v2=0.025
    	v4=0.0145
    	v5=0.008
    	v6=-0.015
    	ca3=4.0e-7
    	ca4=1.5e-7
   	phi=2.664
    	vl=-0.07
    	vk=-0.09
    	vca=0.08
    	gk=3.1416e-13
    	gl=7.854e-14
    	gca=1.57e-13
    	c=1.9635e-14
   	kd= 1.0e-6
    	bt=1.0e-4
    	alpha=7.9976e7
   	kca=1.3567537e2
    
    	p = [v1,v2,v4,v5,v6,ca3,ca4,phi,vl,vk,vca,gk,gl,gca,c,kd,bt,alpha,kca]

    	eq,info,ier,mesg = fsolve(steady, x0, args=(0, p), full_output=1)
	eq_1,info,ier,mesg = fsolve(steady, x1, args=(0, p), full_output=1)
	eq_2,info,ier,mesg = fsolve(steady, x2, args=(0, p), full_output=1)
	eq_3,info,ier,mesg = fsolve(steady, x3, args=(0, p), full_output=1)
	if ier==1:
	   F = np.concatenate((F,eq.reshape(1,3)))
	   P = np.concatenate((P,eq_1.reshape(1,3)))
	   Q = np.concatenate((Q,eq_2.reshape(1,3)))
	   R = np.concatenate((R,eq_3.reshape(1,3)))
	else:
	   F = np.concatenate((F,[[np.nan, np.nan, np.nan]]))
	   P = np.concatenate((P,[[np.nan, np.nan, np.nan]]))
	   Q = np.concatenate((Q,[[np.nan, np.nan, np.nan]]))
           R = np.concatenate((R,[[np.nan, np.nan, np.nan]]))
  	   
        #print(F)
	#print(eq,ier,mesg)
	    
        x=Jacob(eq[0],eq[1],eq[2])
	#print(x)
	eig_val=la.eigvals(x)
	K.append(eig_val)
	#print(K)
	G=np.asarray(K)	
	print(G)
#V=np.asarray(C)
#m=S*np.NAN     
#H=np.concatenate((V,m))


'''
f,ax=plt.subplots()
ax.plot(G.real,G.imag)
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")  
ax.set_title("plot of the eigenvalues")
ax.grid()
'''

'''
f,ax = plt.subplots()
ax.plot(V_1, [f[0] for f in F], 'g')

f,ax = plt.subplots()
ax.plot(V_1, [f[1] for f in F], 'r')

f,ax = plt.subplots()
ax.plot(V_1, [f[2] for f in F], 'b')
'''

#plt.show()

########################################################################################

def func(Y,t,v1,v2,v4,v5,v6,ca3,ca4,phi,vl,vk,vca,gk,gl,gc,c,kd,bt,alpha,kca):
    #n =Y[0], v =Y[1] , ca =Y[2]
    v3=(-v5/2)*np.tanh((Y[2]-ca3)/ca4)+v6
    return np.array([phi*np.cosh((Y[1]-v3)/(2*v4))*(0.5*(1+np.tanh((Y[1]-v3)/v4))-Y[0]),
                     1/c*(-gl*(Y[1]-vl)-gk*Y[0]*(Y[1]-vk)-gc*0.5*(1+np.tanh((Y[1]-v1)/v2))*(Y[1]-vca)),
                    ((-alpha*gc*0.5*(1+np.tanh((Y[1]-v1)/v2))*(Y[1]-vca)-kca*Y[2])*((kd+Y[2])**2/((kd+Y[2])**2+kd*bt)))])

# to generate the x-axes
t= np.linspace(0,10,1000)

#initial values

func0= [0,0, 0]  # [N,V,CA]
func_0=[  3.28788318e-01 , -2.10274713e-02 , -3.94772634e-08]
func_1 = [  4.66449100e-02,  -5.05813969e-02 , -2.96308393e-08]
func_2= [2.44747367e-02 , -2.67980023e-02 , -2.37405136e-08]
func_3= [  2.40141405e-01,  -3.56639256e-02 , -3.90572930e-08]

pars =  (-0.0275,0.025,0.0145,0.008,-0.015,4.0e-7,1.5e-7,2.664,-0.07,-0.09,0.08,3.1416e-13,7.854e-14,1.57e-13,1.9635e-14,1.0e-6,1.0e-4,7.9976e7,1.3567537e2)

Y = integrate.odeint(func, func0, t, pars)

n,v,ca = Y.T

# the plots

'''
plt.subplot(4,1,1)
plt.plot(t, n, 'r', linewidth=2,label='n')
plt.xlabel('t')
plt.ylabel('n(t)')
plt.legend(loc='best')


plt.subplot(4,1,2)
plt.plot(t, v, 'b',linewidth=2, label='v')
plt.xlabel('t')
plt.ylabel('v(t)')
plt.legend(loc='best')


plt.subplot(4,1,3)
plt.plot(t,ca, 'g',linewidth=2, label='ca')
plt.xlabel('t')
plt.ylabel('ca(t)')
plt.legend(loc='best')

plt.subplot(4,1,4)
plt.plot(n,v, 'b',linewidth=2, label='ca')
plt.xlabel('v')
plt.ylabel('n')
plt.legend(loc='best')

'''

######################################################################################################################

# to store the max_min of the solutions
Ymin = []
Ymax = []
Y1min = []
Y1max = []
Y2min = []
Y2max = []
Y3min = []
Y3max = []
Y4min = []
Y4max = []

t = np.linspace(0, 100,1000)

for v1 in V_1:
    pars = (v1,0.025,0.0145,0.008,-0.015,4.0e-7,1.5e-7,2.664,-0.07,-0.09,0.08,3.1416e-13,7.854e-14,1.57e-13,1.9635e-14,1.0e-6,1.0e-4,7.9976e7,1.3567537e2)

    # integrate again the equation, with new parameters
    Y = integrate.odeint(func, func0, t, pars)
    Y1 = integrate.odeint(func,func_0,t,pars)
    Y2 = integrate.odeint(func,func_1,t,pars)
    Y3 = integrate.odeint(func,func_2,t,pars)
    Y4 = integrate.odeint(func,func_3,t,pars)

    # appending the result to the list
    Ymin.append(Y[-60:,:].min(axis=0))
    Ymax.append(Y[-60:,:].max(axis=0))
    Y1min.append(Y1[-60:,:].min(axis=0))
    Y1max.append(Y1[-60:,:].max(axis=0))
    Y2min.append(Y1[-60:,:].min(axis=0))
    Y2max.append(Y1[-60:,:].max(axis=0))
    Y3min.append(Y1[-60:,:].min(axis=0))
    Y3max.append(Y1[-60:,:].max(axis=0))
    Y4min.append(Y1[-60:,:].min(axis=0))
    Y4max.append(Y1[-60:,:].max(axis=0))
# convert the lists into arrays
Ymin = np.asarray(Ymin)
Ymax = np.asarray(Ymax)
Y1min = np.asarray(Y1min)
Y1max = np.asarray(Y1max)
Y2min = np.asarray(Y2min)
Y2max = np.asarray(Y2max)
Y3min = np.asarray(Y3min)
Y3max = np.asarray(Y3max)
Y4min = np.asarray(Y4min)
Y4max = np.asarray(Y4max)

'''
# plot the bifurcation diagram
plt.figure()

plt.subplot(3,1,1)
plt.plot(V_1, Ymin[:,0], 'k', linewidth=2,label='n')
plt.plot(V_1, Ymax[:,0], 'k',linewidth=2)
plt.plot(V_1, Y1min[:,0], 'g*', linewidth=2)
plt.plot(V_1, Y1max[:,0], 'g*',linewidth=2)
plt.plot(V_1, Y2min[:,0], 'b', linewidth=2)
plt.plot(V_1, Y2max[:,0], 'b',linewidth=2)
plt.plot(V_1, Y3min[:,0], 'yo', linewidth=2)
plt.plot(V_1, Y3max[:,0], 'yo',linewidth=2)
plt.plot(V_1, Y4min[:,0], 'r', linewidth=2)
plt.plot(V_1, Y4max[:,0], 'r',linewidth=2)
plt.xlabel('$v1$')
plt.ylabel('n')
plt.legend(loc='best')


plt.subplot(3,1,2)
plt.plot(V_1, Ymin[:,1], 'b',linewidth=2, label='v')
plt.plot(V_1, Ymax[:,1], 'b', linewidth=2)
plt.plot(V_1, Y1min[:,1], 'r', linewidth=2)
plt.plot(V_1, Y1max[:,1], 'r',linewidth=2)
plt.plot(V_1, Y2min[:,1], 'b', linewidth=2)
plt.plot(V_1, Y2max[:,1], 'b',linewidth=2)
plt.plot(V_1, Y3min[:,1], 'y', linewidth=2)
plt.plot(V_1, Y3max[:,1], 'y',linewidth=2)
plt.plot(V_1, Y4min[:,1], 'm--', linewidth=2)
plt.plot(V_1, Y4max[:,1], 'm--',linewidth=2)
plt.xlabel('$v1$')
plt.ylabel('v')
plt.legend(loc='best')


plt.subplot(3,1,3)
plt.plot(V_1, Ymin[:,2], 'g',linewidth=2, label='ca')
plt.plot(V_1, Ymax[:,2], 'g',linewidth=2)
plt.plot(V_1, Y1min[:,2], 'r', linewidth=2)
plt.plot(V_1, Y1max[:,2], 'r',linewidth=2)
plt.plot(V_1, Y2min[:,2], 'b', linewidth=2)
plt.plot(V_1, Y2max[:,2], 'b',linewidth=2)
plt.plot(V_1, Y3min[:,2], 'y', linewidth=2)
plt.plot(V_1, Y3max[:,2], 'y',linewidth=2)
plt.plot(V_1, Y4min[:,2], 'm--', linewidth=2)
plt.plot(V_1, Y4max[:,2], 'm--',linewidth=2)
plt.xlabel('$v1$')
plt.ylabel('ca')
plt.legend(loc='best')

plt.show()
'''
#######################################################################################

f,ax = plt.subplots()
ax.plot(V_1, [f[0] for f in F],'r',V_1, [p[0] for p in P],'m',V_1, [q[0] for q in Q],'b',V_1, [r[0] for r in R],'g', V_1, Ymax[:,0],'b',V_1, Ymin[:,0],'b',V_1, Y1max[:,0], 'g',V_1, Y1min[:,0], 'g',V_1, Y2max[:,0], 'r',V_1, Y2min[:,0], 'r',V_1, Y3max[:,0], 'y',V_1, Y3min[:,0], 'y')

f,ax= plt.subplots()
ax.plot(V_1, [f[1] for f in F],'r',V_1, [p[1] for p in P],'m',V_1, [q[1] for q in Q],'b',V_1, [r[1] for r in R],'g', V_1, Ymax[:,1],'b',V_1, Ymin[:,1],'b',V_1, Y1max[:,1], 'g',V_1, Y1min[:,1], 'g',V_1, Y2max[:,1], 'r',V_1, Y2min[:,1], 'r',V_1, Y3max[:,1], 'y',V_1, Y3min[:,1], 'y')

f,ax= plt.subplots()
ax.plot(V_1, [f[2] for f in F],'r',V_1, [p[2] for p in P],'m',V_1, [q[2] for q in Q],'b',V_1, [r[2] for r in R],'g', V_1, Ymax[:,2],'b',V_1, Ymin[:,2],'b',V_1, Y1max[:,2], 'g',V_1, Y1min[:,2], 'g',V_1, Y2max[:,2], 'r',V_1, Y2min[:,2], 'r',V_1, Y3max[:,2], 'y',V_1, Y3min[:,2], 'y')


plt.show()






