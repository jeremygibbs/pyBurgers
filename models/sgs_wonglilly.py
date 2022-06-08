import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias, Filter

class WongLilly(SGS):
	
	# model class initialization
	def __init__(self,nx,dx):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Wong-Lilly SGS Model")
		
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
		M11   = 2*(self.dx**(4/3))*dudxf*(1-2**(4/3))
		dudx2 = self.dealias.compute(dudx)
		
		if np.mean(M11*M11) == 0:
			cwl = 0
		else:
			cwl = np.mean(L11*M11)/np.mean(M11*M11)
		if cwl < 0: 
			cwl = 0
		
		sgs['tau']   = -2*cwl*(self.dx**(4/3))*dudx2
		-2*CWL*((lam*dx)^(4/3))*dudx;
		sgs['coeff'] = cwl
		
		return self.sgs