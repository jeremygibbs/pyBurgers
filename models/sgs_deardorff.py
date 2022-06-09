import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias, Filter

class Deardorff(SGS):
	
	# model class initialization
	def __init__(self,input):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Deardorff TKE model")
		
		# De-alias object
		self.dealias = Dealias(self.nx)
		
		# Filtering object
		self.filter = Filter(self.nx)
		
		# Derivatives object
		self.derivs = Derivatives(self.nx,self.dx)
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		# Constants
		ce = 0.70
		c1 = 0.10
		
		# dealias to get dudx^2
		dudx2 = self.dealias.compute(dudx)
		
		# compute derivative
		derivs_k  = self.derivs.compute(tke_sgs,[1])
		dkdx      = derivs_k['1']
		
		derivs_ku = self.derivs.compute(tke_sgs*u,[1])
		dkudx     = derivs_ku['1']
		
		Vt  = c1*self.dx*(tke_sgs**0.5)
		tau = -2.*Vt*dudx2
		zz  = 2*Vt*dkdx
		
		derivs_zz = self.derivs.compute(zz,[1])
		dzzdx     = derivs_zz["1"]
		
		dtke    = ((-1*dkudx)+(2*Vt*dudx2*dudx2)+dzzdx-(ce*(tke_sgs**1.5)/self.dx))*self.dt
		tke_sgs = tke_sgs + dtke
		
		self.sgs['tau']     = tau
		self.sgs['coeff']   = c1
		self.sgs['tke_sgs'] = tke_sgs
		
		return self.sgs