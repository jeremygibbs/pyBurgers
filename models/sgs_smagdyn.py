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
		print("[pyBurgers: SGS] \t Using the Dynamic-Coefficient Smagorinsky SGS Model")
		
		# De-alias object
		self.dealias = Dealias(self.nx)
		
		# Filtering object
		self.filter = Filter(self.nx)
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		uf    = self.filter.box(u,2)
		uuf   = self.filter.box(u**2,2)
		L11   = uuf - uf*uf
		dudxf = self.filter.box(dudx,2)
		T     = np.abs(dudx)*dudx
		Tf    = self.filter.box(T,2)   
		M11   = -2*(self.dx**2)*(4*np.abs(dudxf)*dudxf - Tf)
		dudx2 = self.dealias.compute(dudx)
		
		if np.mean(M11*M11) == 0:
			cs2 = 0
		else:
			cs2 = np.mean(L11*M11)/np.mean(M11*M11)
		if cs2 < 0: 
			cs2 = 0
		
		sgs['tau']   = -2*cs2*(self.dx**2)*dudx2
		sgs['coeff'] = np.sqrt(cs2)
		
		return self.sgs