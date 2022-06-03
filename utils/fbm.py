import numpy as np
import pyfftw
from scipy.stats import norm

class FBM(object):
	
	def __init__(self,alpha,n):
		m          = int(n/2)
		k          = np.abs(np.fft.fftfreq(n,d=1/n))
		k[0]       = 1
		x          = pyfftw.empty_aligned(n, dtype='complex128')
		x[:]       = np.sqrt(n)*norm.ppf(np.random.rand(n))
		fx         = pyfftw.interfaces.numpy_fft.fft(x)
		fx[0]      = 0
		fx[m]      = 0
		fx1        = fx * ( k**(-alpha/2) )
		self.noise = np.real(pyfftw.interfaces.numpy_fft.ifft(fx1))
