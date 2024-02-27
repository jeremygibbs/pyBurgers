import multiprocessing
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
		
		# Configure pyfftw
		fftw_nthreads = 4
		fftw_planning = "FFTW_ESTIMATE"
		
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
							   flags=(fftw_planning,),
							   threads=fftw_nthreads)
		
		self.ifft = pyfftw.FFTW(self.fun,
								self.der,
								direction="FFTW_BACKWARD",
								flags=(fftw_planning,),
								threads=fftw_nthreads)
								
		self.fftp = pyfftw.FFTW(self.up,
		                        self.fup,
		                        direction="FFTW_FORWARD",
		                        flags=(fftw_planning,),
		                        threads=fftw_nthreads)
		 
		self.ifftp = pyfftw.FFTW(self.fup,
		                         self.up,
		                         direction="FFTW_BACKWARD",
		                         flags=(fftw_planning,),
		                         threads=fftw_nthreads)
	
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

class Dealias(object):
	
	def __init__(self,nx):
		
		# user values
		self.nx = nx
		
		# computed values
		self.m = int(self.nx/2)
		
		# Configure pyfftw
		fftw_nthreads = 1#multiprocessing.cpu_count()
		fftw_planning = "FFTW_ESTIMATE"
		
		# pyfftw arrays
		self.x  = pyfftw.empty_aligned(self.nx, np.complex128)
		self.fx = pyfftw.empty_aligned(self.nx, np.complex128)
		
		# padded pyfftw arrays
		self.zp   = pyfftw.zeros_aligned(1*self.m, np.complex128)
		self.xp   = pyfftw.empty_aligned(3*self.m, np.complex128)
		self.temp = pyfftw.empty_aligned(3*self.m, np.complex128)
		self.fxp  = pyfftw.empty_aligned(3*self.m, np.complex128)
		
		# pyfftw functions
		self.fft = pyfftw.FFTW(self.x,
							   self.fx,
							   direction="FFTW_FORWARD",
							   flags=(fftw_planning,),
							   threads=fftw_nthreads)
		
		self.ifft = pyfftw.FFTW(self.fx,
								self.x,
								direction="FFTW_BACKWARD",
								flags=(fftw_planning,),
								threads=fftw_nthreads)
								
		self.fftp = pyfftw.FFTW(self.xp,
								self.fxp,
								direction="FFTW_FORWARD",
								flags=(fftw_planning,),
								threads=fftw_nthreads)
		 
		self.ifftp = pyfftw.FFTW(self.fxp,
								 self.xp,
								 direction="FFTW_BACKWARD",
								 flags=(fftw_planning,),
								 threads=fftw_nthreads)
	
	def compute(self,x):
		
		# copy input array
		self.x[:] = x
		
		# compute fft of x 
		self.fft()
		
		# zero-pad fx
		self.fxp[:] = np.concatenate((self.fx[0:self.m+1],self.zp,self.fx[self.m+1:self.nx]))
		
		# compute ifft of fxp
		self.ifftp()
		
		# store xp in temp
		self.temp[:] = self.xp[:]
		
		# change x to abs(x)
		self.x[:] = np.abs(x)
		
		# compute fft of x 
		self.fft()
		
		# zero-pad fx
		self.fxp[:] = np.concatenate((self.fx[0:self.m+1],self.zp,self.fx[self.m+1:self.nx]))
		
		# compute ifft of fxp
		self.ifftp()
		
		# multiply xp[x] with xp[abs(x)]
		self.xp[:] = np.real(self.xp)*np.real(self.temp)
		
		# compute fft of xp
		self.fftp()
		
		# de-alias fxp
		self.fx[:] = np.concatenate((self.fxp[0:self.m+1],self.fxp[2*self.m+1:self.m+self.nx]))
		
		# compute ifft of fx
		self.ifft()
		
		# return de-aliased input
		return (3/2)*np.real(self.x)

class Filter(object):
		
		def __init__(self,nx,nx2=None):
			
			# user values
			self.nx = nx
			
			# Configure pyfftw
			fftw_nthreads = 1#multiprocessing.cpu_count()
			fftw_planning = "FFTW_ESTIMATE"
			
			# pyfftw arrays
			self.x   = pyfftw.empty_aligned(self.nx, np.complex128)
			self.fx  = pyfftw.empty_aligned(self.nx, np.complex128)
			self.fxf = pyfftw.zeros_aligned(self.nx, np.complex128)
			
			# pyfftw functions
			self.fft = pyfftw.FFTW(self.x,
								   self.fx,
								   direction="FFTW_FORWARD",
								   flags=(fftw_planning,),
								   threads=fftw_nthreads)
			
			self.ifft = pyfftw.FFTW(self.fxf,
									self.x,
									direction="FFTW_BACKWARD",
									flags=(fftw_planning,),
									threads=fftw_nthreads)
		
			# check for optional larger nx (for downscaling from DNS->LES)
			if (nx2):
				
				# user value
				self.nx2 = nx2
				
				# pyfftw arrays
				self.x2   = pyfftw.empty_aligned(self.nx2, np.complex128)
				self.fx2  = pyfftw.empty_aligned(self.nx2, np.complex128)
				
				# pyfftw functions
				self.fft2 = pyfftw.FFTW(self.x2,
									    self.fx2,
									    direction="FFTW_FORWARD",
									    flags=("FFTW_ESTIMATE",),
									    threads=4)
		
		# spectral cutoff filter
		def cutoff(self,x,ratio):
			
			# signal size information
			m = int(self.nx/ratio)
			l = int(m/2)
			
			# copy input array
			self.x[:] = x
			
			# compute fft of x
			self.fft()
			
			# filter fx
			self.fxf[0:l] = self.fx[0:l]
			self.fxf[self.nx-l+1:self.nx] = self.fx[self.nx-l+1:self.nx]
			
			# compute ifft of fxf
			self.ifft()
			
			# return filtered x
			return np.real(self.x)
		
		# Fourier filtering from DNS to LES
		def downscale(self,x,ratio):
			
			# copy input array
			self.x2[:] = x
			
			# signal shape information
			l  = int(self.nx/2)
						
			# compute fft of larger series
			self.fft2()
						
			# filter
			self.fxf[l] = 0
			self.fxf[0:l] = self.fx2[0:l]
			self.fxf[l+1:self.nx] = self.fx2[self.nx2-l+1:self.nx2]
			
			# compute the ifft
			self.ifft()
			
			# return filtered downscaled field
			return (1/ratio)*np.real(self.x)