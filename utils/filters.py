#!/usr/bin/env python
import numpy as np 

# Fourier filtering from DNS to LES
def filterDown(self,u,k):
	
	# signal shape information
	n   = int(u.shape[0])
	m   = int(n/k)
	l   = int(m/2)
	
	# compute fft then filter
	fu  = np.fft.fft(u)
	fuf = np.zeros(m,dtype=np.complex)
	fuf[0:l]   = fu[0:l]
	fuf[l+1:m] = fu[n-l+1:n]
	
	# return from spectral space
	uf = (1/k)*np.real(np.fft.ifft(fuf))

	return uf

# Fourier box filter
def filterBox(self,u,k):
	
	# signal size information
	n   = int(u.shape[0])
	m   = int(n/k)
	l   = int(m/2)
	
	# compute fft then filter
	fu  = np.fft.fft(u)
	fuf = np.zeros(n,dtype=np.complex)
	fuf[0:l]     = fu[0:l]
	fuf[n-l+1:n] = fu[n-l+1:n]

	# return from spectral space
	uf = np.real(np.fft.ifft(fuf))

	return uf

# function to de-alias
def dealias1(self,x,n):
	
	# signal size information
	m   = int(n/2)
	
	# compute fft then de-alias
	fx  = np.fft.fft(x)
	fxp = np.concatenate((fx[0:m+1],np.zeros(m),fx[m+1:n]))
	
	# return from spectral space
	xp  = np.real(np.fft.ifft(fxp))
	
	return xp

# function to de-alias
def dealias2(self,xp,n):
	
	# signal size information
	m     = int(n/2)
	
	# compute fft then de-alias
	fxp   = np.fft.fft(xp)
	fx    = np.concatenate((fxp[0:m+1],fxp[2*m+1:m+n]))
	fx[m] = 0
	
	# return from spectral space
	x     = (3/2)*np.real(np.fft.ifft(fx))
	
	return x