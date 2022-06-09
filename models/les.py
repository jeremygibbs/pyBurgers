import multiprocessing
import numpy as np
import pyfftw
import sys
from .sgs import SGS
from utils import FBM, Derivatives, Filter

class LES(object):

    # model class initialization
    def __init__(self,inputObj,outbutObj):
        """Constructor method
        """
        
        # inform users of the simulation type
        print("[pyBurgers: Info] \t You are running in LES mode")
        
        # initialize random number generator
        np.random.seed(1)
        
        # Configure pyfftw
        fftw_nthreads = 1
        fftw_planning = "FFTW_ESTIMATE"
        
        # read configuration variables
        print("[pyBurgers: Setup] \t Reading input settings")
        self.input = inputObj
        self.dt    = self.input.dt
        self.nt    = self.input.nt
        self.visc  = self.input.visc
        self.namp  = self.input.namp
        self.nx    = self.input.nxLES
        self.nxDNS = self.input.nxDNS
        self.model = self.input.sgs
        self.t_save= self.input.t_save
        self.mp    = int(self.nx/2)
        self.dx    = 2*np.pi/self.nx
        
        # fractional browning motion noise instance
        # make same size as DNS
        self.fbm = FBM(0.75,self.nxDNS)
        
        # derivatives object
        self.derivs = Derivatives(self.nx,self.dx)
        
        # filters instance
        self.filter = Filter(self.nx,nx2=self.nxDNS)
        
        # sgs model object
        self.subgrid = SGS.get_model(self.model,self.input)
        
        # grid length
        self.x = np.arange(0,2*np.pi,self.dx)
        
        # set velocity field
        self.u    = pyfftw.empty_aligned(self.nx,dtype='complex128')
        self.fu   = pyfftw.empty_aligned(self.nx,dtype='complex128')
        self.fft  = pyfftw.FFTW(self.u,
                                self.fu,
                                direction='FFTW_FORWARD',
                                flags=(fftw_planning,),
                                threads=fftw_nthreads)
        self.ifft = pyfftw.FFTW(self.fu,
                                self.u,
                                direction='FFTW_BACKWARD',
                                flags=(fftw_planning,),
                                threads=fftw_nthreads)
        
        # initialize subgrid tke if using Deardorff SGS model
        if self.model==4:
            self.tke_sgs = np.ones(self.nx)
        else: 
            self.tke_sgs = 0
        
        # other fields
        self.tke      = np.zeros(1)
        self.C_sgs    = np.zeros(1)
        self.diss_sgs = np.zeros(1)
        self.diss_mol = np.zeros(1)
        self.ens_prod = np.zeros(1)
        self.ens_dsgs = np.zeros(1)
        self.ens_dmol = np.zeros(1)
        
        # output
        self.output = outbutObj
        self.output_dims = {
            't' : 0,
            'x' : self.nx
            }
        self.output.set_dims(self.output_dims)
        
        # set reference to output fields
        self.output_fields = {
            'x'            : self.x,
            'u'            : self.u,
            'tke'          : self.tke,
            'C_sgs'        : self.C_sgs,
            'diss_sgs'     : self.diss_sgs,
            'diss_mol'     : self.diss_mol,
            'ens_prod'     : self.ens_prod,
            'ens_diss_sgs' : self.ens_dsgs,
            'ens_diss_mol' : self.ens_dmol
        }
        # add subgrid tke if using Deardorff model
        if self.model==4:
            self.output_fields['tke_sgs'] = self.tke_sgs
        
        self.output.set_fields(self.output_fields)
        
        # write initial data
        self.output.save(self.output_fields,0,0,initial=True)
        
    def run(self):
        
        # placeholder 
        rhsp = 0
        
        # time loop
        for t in range(1,30001):
            
            # get current time
            looptime = t*self.dt
            sys.stdout.write("\r[PyBurgers: Run] \t Running for time %05.2f of %05.2f"%(looptime,int(self.nt)*self.dt))
            sys.stdout.flush()
            
            # compute derivative
            # if save time, compute additional derivative
            if (t%self.t_save==0):
                derivatives = self.derivs.compute(self.u,[1,2,3,'sq'])
                d3udx3      = derivatives['3']
            else:
                derivatives = self.derivs.compute(self.u,[1,2,'sq'])
            
            dudx   = derivatives['1']
            du2dx  = derivatives['sq']
            d2udx2 = derivatives['2']
            
            # add fractional Brownian motion (FBM) noise
            # then filter noise from DNS to LES scales
            noise = self.fbm.compute_noise()
            noise = self.filter.downscale(noise,int(self.nxDNS/self.nx))
            
            # compute subgrid terms
            sgs    = self.subgrid.compute(self.u, dudx, self.tke_sgs)
            tau    = sgs["tau"]
            coeff  = sgs["coeff"]

            if self.model==4:
                self.tke_sgs[:] = sgs["tke_sgs"]
            sgsder = self.derivs.compute(tau,[1])
            dtaudx   = sgsder['1']
            
            # compute right hand side
            rhs = self.visc * d2udx2 - 0.5*du2dx + np.sqrt(2*self.namp/self.dt)*noise - 0.5*dtaudx
            
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
                
                # time index
                t_out = int(t/self.t_save)
                
                # turbulence kinetic energy
                self.tke[:] = np.var(self.u)
                
                # dissipation
                self.diss_sgs[:] = np.mean(-tau*dudx)
                self.diss_mol[:] = np.mean(self.visc*dudx**2)
                
                # enstrophy
                self.ens_prod[:] = np.mean(dudx**3)
                self.ens_dsgs[:] = np.mean(-tau*d3udx3)
                self.ens_dmol[:] = np.mean(self.visc*d2udx2**2)
                
                # sgs coefficient
                self.C_sgs[:] = coeff
                
                # save fields
                self.output.save(self.output_fields,t_out,looptime,initial=False)
        
        # close output file       
        self.output.close()