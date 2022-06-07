import sys
import time
import cmath as cm
import netCDF4 as nc
import numpy as np
import pyfftw

class Derivatives(object):

	def __init__(self,nx,dx):
		
		# user values
		self.nx = nx
		self.dx = dx
		
		# computed values
		self.m         = int(self.nx/2)
		self.fac       = 2*np.pi/(self.nx*self.dx)
		self.k         = np.abs(np.fft.fftfreq(self.nx,d=1/self.nx))
		self.k[self.m] = 0
		
		# pyfftw arrays
		self.u   = pyfftw.empty_aligned(nx, np.complex128)
		self.fu  = pyfftw.empty_aligned(nx, np.complex128)
		self.fun = pyfftw.empty_aligned(nx, np.complex128)
		self.der = pyfftw.empty_aligned(nx, np.complex128)
		
		# padded pyfftw arrays
		self.zp  = pyfftw.zeros_aligned(nx, np.complex128)
		self.up  = pyfftw.empty_aligned(nx*2, np.complex128)
		self.up2 = pyfftw.empty_aligned(nx*2, np.complex128)
		self.fup = pyfftw.empty_aligned(nx*2, np.complex128)
		
		# pyfftw functions
		self.fft = pyfftw.FFTW(self.u,
							   self.fu,
							   direction="FFTW_FORWARD",
							   flags=("FFTW_ESTIMATE",),
							   threads=1)
		
		self.ifft = pyfftw.FFTW(self.fun,
								self.der,
								direction="FFTW_BACKWARD",
								flags=("FFTW_ESTIMATE",),
								threads=1)
		self.fftp = pyfftw.FFTW(self.up,
		                        self.fup,
		                        direction="FFTW_FORWARD",
		                        flags=("FFTW_ESTIMATE",),
		                        threads=1)
		 
		self.ifftp = pyfftw.FFTW(self.fup,
		                         self.up,
		                         direction="FFTW_BACKWARD",
		                         flags=("FFTW_ESTIMATE",),
		                         threads=1)
	
	def compute(self,u,order):
		
		# return dictionary
		derivatives = {}
		
		# copy input array
		self.u[:] = u
		
		# compute fft
		self.fft()
		
		# loop through order of derivative from user
		for key in order:
			
			if key==1:
				self.fun[:] = cm.sqrt(-1)*self.k*self.fu
				self.ifft()
				derivatives['1'] = self.fac*np.real(self.der)
			if key==2:
				self.fun[:] = -self.k*self.k*self.fu
				self.ifft()
				derivatives['2'] = self.fac**2*np.real(self.der)
			if key==3:
				self.fun[:] = -cm.sqrt(-1)*self.k**3*self.fu
				self.ifft()
				derivatives['3'] = self.fac**3*np.real(self.der)
			if key=='sq':
				self.fup[:] = np.insert(self.fu,self.m,self.zp)
				self.ifftp()
				self.up[:] = self.up[:]**2
				self.fftp()
				self.fu[0:self.m] = self.fup[0:self.m]
				self.fu[self.m::] = self.fup[self.nx+self.m:]
				self.fun[:] = cm.sqrt(-1)*self.k*self.fu
				self.ifft()
				derivatives['sq'] = 2*self.fac*np.real(self.der)
	
		return derivatives