#/usr/bin/env python
import numpy as np
import cmath as cm

def computeDerivative(u,dx,opt=1):
	
	n = np.int(u.shape[0])
	m = int(n/2)
	
	# Fourier colocation method
	if opt==1:
		
		h       = 2*np.pi/n
		fac     = h/dx
		k       = np.fft.fftfreq(n,d=1/n)
		fu      = np.fft.fft(u)
		
		dudx    = fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu))
		d2udx2  = fac**2 * np.real(np.fft.ifft(-k*k*fu))
		d3udx3  = fac**3 * np.real(np.fft.ifft(-cm.sqrt(-1)*k**3*fu))

		# dealiasing needed for du2dx using zero-padding 
		zeroPad = np.zeros(n)
		fu_p    = np.insert(fu,m,zeroPad)
		u_p     = np.real(np.fft.ifft(fu_p))		
		u2_p    = u_p**2
		fu2_p   = np.fft.fft(u2_p)
		fu2     = fu2_p[0:m]
		fu2     = np.append(fu2,fu2_p[n+m:])
		du2dx   = 2*fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu2))
		
		return [dudx,d2udx2,du2dx,d3udx3]