import multiprocessing
import numpy as np
import pyfftw
from scipy.stats import norm

class FBM(object):
	
	def __init__(self,alpha,n):
		
		# user values
		self.n = n
		self.a = alpha
		
		# computed values
		self.m    = int(n/2)
		self.k    = np.abs(np.fft.fftfreq(n,d=1/n))
		self.k[0] = 1
		
		# Configure pyfftw
		fftw_nthreads = 4
		fftw_planning = "FFTW_ESTIMATE"
		
		# pyfftw arrays
		self.x     = pyfftw.empty_aligned(n, np.complex128)
		self.fx    = pyfftw.empty_aligned(n, np.complex128)
		self.fxn   = pyfftw.empty_aligned(n, np.complex128)
		self.noise = pyfftw.empty_aligned(n, np.complex128)
		
		# pyfftw functions
		self.fft = pyfftw.FFTW(self.x,
							   self.fx,
							   direction="FFTW_FORWARD",
		                       flags=(fftw_planning,),
		                       threads=fftw_nthreads)
		
		self.ifft = pyfftw.FFTW(self.fxn,
		                        self.noise,
		                        direction="FFTW_BACKWARD",
		                        flags=(fftw_planning,),
		                        threads=fftw_nthreads)
		
	def compute_noise(self):
		
		# compute input
		self.x[:] = np.sqrt(self.n)*norm.ppf(np.random.rand(self.n))
		
		# compute fft
		self.fft()
		
		# zero-out first and nyquist
		self.fx[0]      = 0
		self.fx[self.m] = 0
		self.fxn[:]     = self.fx * ( self.k**(-self.a/2) )
		self.ifft()
		
		return(np.real(self.noise))