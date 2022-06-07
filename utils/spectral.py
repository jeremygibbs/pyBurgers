import sys
import time
import netCDF4 as nc
import numpy as np
import pyfftw

class Spectral(object):

	def __init__(self):
		
		
	
	# function to compute spatial derivatives in spectral space
	def derivative(self,u,dx):
		
		# signal shape information
		n = int(u.shape[0])
		m = int(n/2)
		
		# Fourier colocation method
		h       = 2*np.pi/n
		fac     = h/dx
		k       = np.fft.fftfreq(n,d=1/n)
		k[m]    = 0
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
	
		# store derivatives in a dictionary for selective access
		derivatives = {
			'dudx'  :   dudx,
			'du2dx' :   du2dx,
			'd2udx2':   d2udx2,
			'd3udx3':   d3udx3
		}
		
		return derivatives
	
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