import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias

class SmagConstant(SGS):
	
	# model class initialization
	def __init__(self,input):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Constant-Coefficient Smagorinsky SGS Model")
		
		# De-alias object
		self.dealias = Dealias(self.nx)
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		cs2   = 0.16**2
		dudx2 = self.dealias.compute(dudx)
		
		sgs['tau']   = -2*cs2*(self.dx**2)*dudx2
		sgs['coeff'] = np.sqrt(cs)
		
		return self.sgs