# /usr/bin/env python
import numpy as np

def addNoise(alpha,n):
	
	x     = np.sqrt(n)*np.random.randn(n);
	m     = int(n/2)
	k     = np.abs(np.fft.fftfreq(n,d=1/n))
	k[0]  = 1
	fx    = np.fft.fft(x)
	fx[0] = 0
	fx[m] = 0
	fx1   = fx * ( k**(-alpha/2) )
	x1    = np.real(np.fft.ifft(fx1) )
	
	return x1