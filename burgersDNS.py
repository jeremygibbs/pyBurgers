#/usr/bin/env python
import pylab as pl
import numpy as np
import burgersDerivatives as bd
import burgersFBM as fbm
import sys

# Input parameters
nx   = 8192
dx   = 2*np.pi/nx
dt   = 1E-4
nt   = 2E2
visc = 1E-5
diff = 1E-6

# initialize velocity field
u      = np.ones(nx)
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
	if t == 1:
		u_new = u + dt*rhs
	else:
		u_new = u + dt*(1.5*rhs - 0.5*rhsp)
	
	fu_new     = np.fft.fft(u_new)
	fu_new[mp] = 0
	u_new      = np.real(np.fft.ifft(fu_new))
	u          = u_new
	rhsp       = rhsp