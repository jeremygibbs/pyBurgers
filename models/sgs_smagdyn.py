import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias, Filter

class SmagDynamic(SGS):
	
	# model class initialization
	def __init__(self,input):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Dynamic Smagorinsky model")
		
		# De-alias object
		self.dealias = Dealias(self.nx)
		
		# Filtering object
		self.filter = Filter(self.nx)
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		# L11 term
		uf    = self.filter.cutoff(u,2)
		uuf   = self.filter.cutoff(u**2,2)
		L11   = uuf - uf*uf
		
		# M11 term
		dudxf = self.filter.cutoff(dudx,2)
		T     = np.abs(dudx)*dudx
		Tf    = self.filter.cutoff(T,2)   
		M11   = (self.dx**2)*(4*np.abs(dudxf)*dudxf - Tf)
		dudx2 = self.dealias.compute(dudx)
		
		# Smagorinsky coefficient
		if np.mean(M11*M11) == 0:
			cs2 = 0
		else:
			cs2 = 0.5 * np.mean(L11*M11)/np.mean(M11*M11)
			if cs2 < 0: 
				cs2 = 0
		
		# set sgs dictionary
		self.sgs['tau']   = -2*cs2*(self.dx**2)*dudx2
		self.sgs['coeff'] = np.sqrt(cs2)
		
		return self.sgs