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
            
            rhsp = rhs
            
            # write output
            if (t%self.t_save==0):
                t_out       = int(t/self.t_save)
                self.tke[:] = np.var(self.u)
                self.output.save(self.output_fields,t_out,looptime,initial=False)
                
        # close output file       
        self.output.close()

# 
# #/usr/bin/env python
# import time
# from sys import stdout
# import numpy as np
# import netCDF4 as nc
# import pyfftw
# from burgers import Utils, Settings
# 
# 
# 
# # DNS run loop
# def main():
# 
#     # let's time this thing
#     t1 = time.time()
# 
#     # a nice welcome message
#     print("##############################################################")
#     print("#                                                            #")
#     print("#                   Welcome to pyBurgers                     #")
#     print("#      A fun tool to study turbulence using DNS and LES      #")
#     print("#                                                            #")
#     print("##############################################################")
#     print("[pyBurgers: Info] \t You are running in DNS mode")
# 
#     # instantiate helper classes
#     print("[pyBurgers: Setup] \t Reading input settings")
#     utils    = Utils()
#     settings = Settings('namelist.json')
# 
#     # input settings
#     nx   = settings.nxDNS
#     mp   = int(nx/2)
#     dx   = 2*np.pi/nx
#     dt   = settings.dt
#     nt   = settings.nt
#     visc = settings.visc
#     damp = settings.damp
#     
#     # initialize velocity field
#     print("[pyBurgers: Setup] \t Initialzing velocity field")
#     #u = np.zeros(nx)
#     u  = pyfftw.empty_aligned(nx,dtype='float64')
#     fu = pyfftw.empty_aligned(nx//2+1,dtype='complex128')
#     fft_object = pyfftw.FFTW(u,fu,direction='FFTW_FORWARD')
#     ifft_object = pyfftw.FFTW(fu,u,direction='FFTW_BACKWARD')
#     
#     # initialize random number generator
#     np.random.seed(1)
# 
#     # place holder for right hand side
#     rhsp = 0
#   
#     # create output file
#     print("[pyBurgers: Setup] \t Creating output file")
#     output = nc.Dataset('pyBurgersDNS.nc','w')
#     output.description = "pyBurgers DNS output"
#     output.source = "Jeremy A. Gibbs"
#     output.history = "Created " + time.ctime(time.time())
#     
#     # add dimensions
#     output.createDimension('t')
#     output.createDimension('x',nx)
# 
#     # add variables
#     out_t = output.createVariable("t", "f4", ("t"))
#     out_t.long_name = "time"
#     out_t.units = "s"
#     out_x = output.createVariable("x", "f4", ("x"))
#     out_x.long_name = "x-distance"
#     out_x.units = "m"
#     out_k = output.createVariable("tke", "f4", ("t"))
#     out_k.long_name = "turbulence kinetic energy"
#     out_k.units = "m2 s-2"
#     out_u = output.createVariable("u", "f4", ("t","x"))
#     out_u.long_name = "velocity"
#     out_u.units = "m s-1"
# 
#     # write x data
#     out_x[:] = np.arange(0,2*np.pi,dx)
# 
#     # time loop
#     save_t = 0
#     for t in range(int(nt)):
# 
#         # update progress
#         if (t==0 or (t+1)%1000==0):
#             stdout.write("\r[pyBurgers: DNS] \t Running for time %07d of %d"%(t+1,int(nt)))
#             stdout.flush()
#         
#         # compute derivatives
#         derivs = utils.derivative(u,dx)
#         du2dx  = derivs['du2dx']
#         d2udx2 = derivs['d2udx2'] 
# 
#         # add fractional Brownian motion (FBM) noise
#         fbm = utils.noise(0.75,nx)
# 
#         # compute right hand side
#         rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*fbm
#         
#         # time integration
#         if t == 0:
#             # Euler for first time step
#             u = u + dt*rhs
#         else:
#             # 2nd-order Adams-Bashforth
#             u = u + dt*(1.5*rhs - 0.5*rhsp)
#         
#        # uc = u.copy()
#         
#         # set Nyquist to zero
#         fft_object()
#         fu[mp] = 0
#         ifft_object()
#         
#         # test = np.fft.fft(uc)
#         # test[mp] = 0
#         # un = np.real(np.fft.ifft(test))
#         # 
#         # close = np.allclose(u,un)
#         # print(close)
#         #fu_new     = np.fft.fft(u_new)
#         #fu_new[mp] = 0
#         #u_new      = np.real(np.fft.ifft(fu_new))
#         #u          = u_new
#         rhsp       = rhs
# 
#         # output to file every 1000 time steps (0.1 seconds)
#         if ((t+1)%1000==0):          
#             
#             # kinetic energy
#             tke  = 0.5*np.var(u)
#             
#             # save to disk
#             out_t[save_t]   = (t+1)*dt 
#             out_k[save_t]   = tke
#             out_u[save_t,:] = np.real(u)
#             save_t += 1
# 
#     # time info
#     t2 = time.time()
#     tt = t2 - t1
#     print("\n[pyBurgers: DNS] \t Done! Completed in %0.2f seconds"%tt)
#     print("##############################################################")
# 
# 
# if __name__ == "__main__":
#     main()