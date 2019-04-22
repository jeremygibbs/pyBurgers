#/usr/bin/env python
import pylab as pl
import numpy as np
import burgersDerivatives as bd
import burgersFBM as fbm
import sys

# input parameters
nx   = 8192
dx   = 2*np.pi/nx
dt   = 1E-4
nt   = 2E6
visc = 1E-5
diff = 1E-6

# initialize velocity field
u      = np.zeros(nx)
fu     = np.fft.fft(u) 
mp     = int(nx/2)
fu[mp] = 0
u      = np.real(np.fft.ifft(fu))

# initialize random number generator
np.random.seed(0)

# place holder
rhsp = 0

# advance in time
for t in range(int(nt)):
	
	# compute derivatives
	dudx,d2udx2,du2dx,d3udx3 = bd.computeDerivative(u,dx,1)
	
	# add fractional Brownian motion (FBM) noise
	f = fbm.addNoise(0.75,nx)
	
	# compute right hand side
	rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*diff/dt)*f
	if t == 0:
		# Euler
		u_new = u + dt*rhs
	else:
		# Adams-Bashforth 2nd
		u_new = u + dt*(1.5*rhs - 0.5*rhsp)
	
	fu_new     = np.fft.fft(u_new)
	fu_new[mp] = 0
	u_new      = np.real(np.fft.ifft(fu_new))
	u          = u_new
	rhsp       = rhs

	# output to screen every 100 time steps
	if (t%100==0):
		CFL = np.max(np.abs(u))*dt/dx          
		KE	= 0.5*np.var(u)
		print("%d \t %f \t %f \t %f \t %f \t %f"%(t,t*dt,KE,CFL,np.max(u),np.min(u)))