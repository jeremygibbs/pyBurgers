import numpy as np
from .burgers import Burgers

class LES(Burgers):

    # child class initialization
    def __init__(self,inputObj):
        
        # inform users of the simulation type
        print("[pyBurgers: Info] \t You are running in LES mode")
        
        # initialize parent class
        super().__init__(inputObj)
        
        # read configuration variables
        print("[pyBurgers: Setup] \t Reading input settings")
        self.nx = self.input.nxLES
        self.mp   = int(self.nx/2)
        self.dx   = 2*np.pi/self.nx
        
    def run(self):
        print("LES ran")
        print(self.dt,self.visc)
    
# #/usr/bin/env python
# import time
# from sys import stdout
# import numpy as np
# import netCDF4 as nc
# from burgers import Utils, Settings, BurgersLES
# 
# # LES run loop
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
#     print("[pyBurgers: Info] \t You are running in LES mode")
# 
#     # instantiate helper classes
#     print("[pyBurgers: Setup] \t Reading input settings")
#     utils    = Utils()
#     settings = Settings('namelist.json')
#     
#     # input settings
#     nxDNS = settings.nxDNS
#     nxLES = settings.nxLES
#     mp    = int(nxLES/2)
#     dx    = 2*np.pi/nxLES
#     dt    = settings.dt
#     nt    = settings.nt
#     model = settings.sgs
#     visc  = settings.visc
#     damp  = settings.damp
# 
#     # intantiate the LES SGS class
#     LES = BurgersLES(model)
#     
#     # initialize velocity field
#     print("[pyBurgers: Setup] \t Initialzing velocity field")
#     u = np.zeros(nxLES)
# 
#     # initialize subgrid tke if using Deardorff
#     if model==4:
#         kr = np.ones(nxLES)
#     else:
#         kr = 0
# 
#     # initialize random number generator
#     np.random.seed(1)
# 
#     # place holder for right hand side
#     rhsp = 0
#     
#     # create output file
#     print("[pyBurgers: Setup] \t Creating output file")
#     output = nc.Dataset('pyBurgersLES.nc','w')
#     output.description = "pyBurgers LES output"
#     output.source = "Jeremy A. Gibbs"
#     output.history = "Created " + time.ctime(time.time())
#     output.setncattr("sgs","%d"%model)
#     
#     # add dimensions
#     output.createDimension('t')
#     output.createDimension('x',nxLES)
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
#     out_c = output.createVariable("C", "f4", ("t"))
#     out_c.long_name = "subgrid model coefficient"
#     out_c.units = "--"
#     out_ds = output.createVariable("diss_sgs", "f4", ("t"))
#     out_ds.long_name = "subgrid dissipation"
#     out_ds.units = "m2 s-3"
#     out_dm = output.createVariable("diss_mol", "f4", ("t"))
#     out_dm.long_name = "molecular dissipation"
#     out_dm.units = "m2 s-3"
#     out_ep = output.createVariable("ens_prod", "f4", ("t"))
#     out_ep.long_name = "enstrophy production"
#     out_ep.units = "s-3"
#     out_eds = output.createVariable("ens_diss_sgs", "f4", ("t"))
#     out_eds.long_name = "subgrid enstrophy dissipation"
#     out_eds.units = "s-3"
#     out_edm = output.createVariable("ens_diss_mol", "f4", ("t"))
#     out_edm.long_name = "molecular enstrophy dissipation"
#     out_edm.units = "s-3"
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
#             stdout.write("\r[pyBurgers: LES] \t Running for time %07d of %d"%(t+1,int(nt)))
#             stdout.flush()
#         
#         # compute derivatives
#         derivs = utils.derivative(u,dx)
#         dudx   = derivs['dudx']
#         du2dx  = derivs['du2dx']
#         d2udx2 = derivs['d2udx2']
#         d3udx3 = derivs['d3udx3']
# 
#         # add fractional Brownian motion (FBM) noise
#         fbm  = utils.noise(0.75,nxDNS)
#         fbmf = utils.filterDown(fbm,int(nxDNS/nxLES))
# 
#         # compute subgrid terms
#         sgs    = LES.subgrid(u,dudx,dx,kr)
#         tau    = sgs["tau"]
#         coeff  = sgs["coeff"]
#         if model==4:
#             kr = sgs["kr"]
#         dtaudx = utils.derivative(tau,dx)["dudx"]
# 
#         # compute right hand side
#         rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*fbmf - 0.5*dtaudx
#         
#         # time integration
#         if t == 0:
#             # Euler for first time step
#             u_new = u + dt*rhs
#         else:
#             # 2nd-order Adams-Bashforth
#             u_new = u + dt*(1.5*rhs - 0.5*rhsp)
#         
#         # set Nyquist to zero
#         fu_new     = np.fft.fft(u_new)
#         fu_new[mp] = 0
#         u_new      = np.real(np.fft.ifft(fu_new))
#         u          = u_new
#         rhsp       = rhs
# 
#         # output to file every 1000 time steps (0.1 seconds)
#         if ((t+1)%1000==0):
#             
#             # kinetic energy
#             tke  = 0.5*np.var(u)
# 
#             # dissipation
#             diss_sgs = np.mean(-tau*dudx)
#             diss_mol = np.mean(visc*dudx**2)
# 
#             # enstrophy
#             ens_prod = np.mean(dudx**3)
#             ens_dsgs = np.mean(-tau*d3udx3)
#             ens_dmol = np.mean(visc*d2udx2**2)
#             
#             # save to disk
#             out_t[save_t]   = (t+1)*dt 
#             out_k[save_t]   = tke
#             out_c[save_t]   = coeff
#             out_ds[save_t]  = diss_sgs
#             out_dm[save_t]  = diss_mol
#             out_ep[save_t]  = ens_prod
#             out_eds[save_t] = ens_dsgs
#             out_edm[save_t] = ens_dmol
#             out_u[save_t,:] = u
#             save_t += 1
#     
#     # time info
#     t2 = time.time()
#     tt = t2 - t1
#     print("\n[pyBurgers: LES] \t Done! Completed in %0.2f seconds"%tt)
#     print("##############################################################")
# 
# if __name__ == "__main__":
#     main()