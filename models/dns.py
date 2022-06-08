import numpy as np
import pyfftw
import sys
from utils import FBM, Derivatives

class DNS(object):

    # model class initialization
    def __init__(self,inputObj,outbutObj):
        """Constructor method
        """
        
        # inform users of the simulation type
        print("[pyBurgers: Info] \t You are running in DNS mode")
        
        # initialize random number generator
        np.random.seed(1)
        
        # read configuration variables
        print("[pyBurgers: Setup] \t Reading input settings")
        self.input = inputObj
        self.dt    = self.input.dt
        self.nt    = self.input.nt
        self.visc  = self.input.visc
        self.namp  = self.input.namp
        self.nx    = self.input.nxDNS
        self.t_save= self.input.t_save
        self.mp    = int(self.nx/2)
        self.dx    = 2*np.pi/self.nx
        
        # fractional browning motion noise instance
        self.fbm = FBM(0.75,self.nx)
        
        # derivatives object
        self.derivs = Derivatives(self.nx,self.dx)
        
        # grid length
        self.x = np.arange(0,2*np.pi,self.dx)
        
        # set velocity field
        self.u    = pyfftw.empty_aligned(self.nx,dtype='complex128')
        self.fu   = pyfftw.empty_aligned(self.nx,dtype='complex128')
        self.fft  = pyfftw.FFTW(self.u,self.fu,direction='FFTW_FORWARD')
        self.ifft = pyfftw.FFTW(self.fu,self.u,direction='FFTW_BACKWARD')
        
        # other fields
        self.tke = np.zeros(1)
        
        # output
        self.output = outbutObj
        self.output_dims = {
            't' : 0,
            'x' : self.nx
        	}
        self.output.set_dims(self.output_dims)
        
        # set reference to output fields
        self.output_fields = {
        	'x'   : self.x,
        	'u'   : self.u,
        	'tke' : self.tke,
        }
        self.output.set_fields(self.output_fields)
        
        # write initial data
        self.output.save(self.output_fields,0,0,initial=True)
        
    # main run loop  
    def run(self):
        
        # placeholder 
        rhsp = 0
        
        # time loop
        for t in range(1,10001):
            
            # get current time
            looptime = t*self.dt
            sys.stdout.write("\r[PyBurgers: Run] \t Running for time %05.2f of %05.2f"%(looptime,int(self.nt)*self.dt))
            sys.stdout.flush()
            
            # compute derivative
            derivatives = self.derivs.compute(self.u,[2,'sq'])
            d2udx2 = derivatives['2']
            du2dx  = derivatives['sq']
            
            # add fractional Brownian motion (FBM) noise
            noise = self.fbm.compute_noise()
            
            # compute right hand side
            rhs = self.visc * d2udx2 - 0.5*du2dx + np.sqrt(2*self.namp/self.dt)*noise
            
            # time integration
            if t == 0:
                # Euler for first time step
                self.u[:] = self.u[:] + self.dt*rhs
            else:
                # 2nd-order Adams-Bashforth
                self.u[:] = self.u[:] + self.dt*(1.5*rhs - 0.5*rhsp)
            
            # set Nyquist to zero
            self.fft()
            self.fu[self.mp] = 0
            self.ifft()
            
            # set placeholder rhs to current value
            rhsp = rhs
            
            # write output
            if (t%self.t_save==0):
                t_out       = int(t/self.t_save)
                self.tke[:] = np.var(self.u)
                self.output.save(self.output_fields,t_out,looptime,initial=False)
                
        # close output file       
        self.output.close()