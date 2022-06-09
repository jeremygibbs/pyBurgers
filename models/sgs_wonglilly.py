import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias, Filter

class WongLilly(SGS):
	
	# model class initialization
	def __init__(self,input):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Wong-Lilly model")
		
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
		M11   = self.dx**(4/3) * (1-2**(4/3)) * dudxf
		
		# Wong-Lilly coefficient		
		if np.mean(M11*M11) == 0:
			cwl = 0
		else:
			cwl = 0.5 * np.mean(L11*M11)/np.mean(M11*M11)
			if cwl < 0: 
				cwl = 0
		
		# set sgs dictionary		
		self.sgs['tau']   = -2*cwl*(self.dx**(4/3))*dudx
		self.sgs['coeff'] = cwl
		
		return self.sgs