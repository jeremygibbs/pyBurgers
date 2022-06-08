import numpy as np
from .sgs import SGS
from utils import FBM, Derivatives, Dealias, Filter

class Deardorff(SGS):
	
	# model class initialization
	def __init__(self,nx,dx):
		"""Constructor method
		"""
		
		# initialize parent class
		super().__init__(input)
		
		# inform users of the sgs model type
		print("[pyBurgers: SGS] \t Using the Deardorff TKE SGS Model")
		
		# De-alias object
		self.dealias = Dealias(self.nx)
		
		# Filtering object
		self.filter = Filter(self.nx)
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		ce = 0.70
		c1 = 0.10
		
		
		uf    = self.filter.box(u,2)
		uuf   = self.filter.box(u**2,2)
		L11   = uuf - uf*uf
		dudxf = self.filter.box(dudx,2)
		M11   = 2*(self.dx**(4/3))*dudxf*(1-2**(4/3))
		
		if np.mean(M11*M11) == 0:
			cwl = 0
		else:
			cwl = np.mean(L11*M11)/np.mean(M11*M11)
		if cwl < 0: 
			cwl = 0
		
		sgs['tau']   = -2*cwl*(self.dx**(4/3))*self.dealias.compute(dudx)
		sgs['coeff'] = cwl
		
		
		dt = settings.dt                                  
		d1 = utils.dealias1(np.abs(dudx),n)
		d2 = utils.dealias1(dudx,n)
		d3 = utils.dealias2(d1*d2,n)
		
		derivs_kru = utils.derivative(u*kr,dx)
		derivs_kr  = utils.derivative(kr,dx)
		dukrdx     = derivs_kru["dudx"]
		dkrdx      = derivs_kr["dudx"]
		
		Vt  = C1*dx*(kr**0.5)
		tau = -2.*Vt*d3
		zz  = 2*Vt*dkrdx
		
		derivs_zz = utils.derivative(zz,dx)
		dzzdx     = derivs_zz["dudx"]
		
		dkr  = ( (-1*dukrdx) + (2*Vt*d3*d3) + dzzdx - (Ce*(kr**1.5)/dx) ) * dt
		kr   = kr + dkr
		coeff = C1
		
		sgs = {
			'tau'   :   tau,
			'coeff' :   coeff,
			'kr'    :   kr
		}
		return sgs
		
		return self.sgs