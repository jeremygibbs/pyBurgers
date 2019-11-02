#/usr/bin/env python
import pylab as pl
import numpy as np
import netCDF4 as nc
from burgers import Utils, Settings, BurgersLES

# LES run loop
def main():

    # instantiate helper classes
    utils    = Utils()
    settings = Settings('namelist.json')
    
    # input settings
    nxDNS = settings.nxDNS
    nxLES = settings.nxLES
    mp    = int(nxLES/2)
    dx    = 2*np.pi/nxLES
    dt    = settings.dt
    nt    = settings.nt
    model = settings.sgs
    visc  = settings.visc
    damp  = settings.damp

    # intantiate the LES SGS class
    LES = BurgersLES(model)
    
    # initialize velocity field
    u = np.zeros(nxLES)

    # initialize subgrid tke if using Deardorff
    if model==4:
        kr = np.ones(nxLES)
    else:
        kr = 0

    # initialize random number generator
    np.random.seed(1)

    # place holder for right hand side
    rhsp = 0
    
    # create output file
    output = nc.Dataset('pyBurgersLES.nc','w')
    
    # add dimensions
    output.createDimension('t')
    output.createDimension('x',nxLES)

    # add variables
    out_t = output.createVariable("t", "f4", ("t"))
    out_x = output.createVariable("x", "f4", ("x"))
    out_u = output.createVariable("u", "f4", ("t","x"))

    # write x data
    out_x[:] = np.arange(0,2*np.pi,dx)
 
    # time loop
    save_t = 0
    for t in range(int(nt)):
        
        # compute derivatives
        derivs = utils.derivative(u,dx)
        dudx   = derivs['dudx']
        du2dx  = derivs['du2dx']
        d2udx2 = derivs['d2udx2']
        d3udx3 = derivs['d3udx3']

        # add fractional Brownian motion (FBM) noise
        fbm  = utils.noise(0.75,nxDNS)
        fbmf = utils.filterDown(fbm,int(nxDNS/nxLES))

        # compute subgrid terms
        sgs    = LES.subgrid(u,dudx,dx,kr)
        tau    = sgs["tau"]
        coeff  = sgs["coeff"]
        if model==4:
            kr = sgs["kr"]
        dtaudx = utils.derivative(tau,dx)["dudx"]

        # compute right hand side
        rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*fbmf - 0.5*dtaudx
        
        # time integration
        if t == 0:
            # Euler for first time step
            u_new = u + dt*rhs
        else:
            # 2nd-order Adams-Bashforth
            u_new = u + dt*(1.5*rhs - 0.5*rhsp)
        
        # set Nyquist to zero
        fu_new     = np.fft.fft(u_new)
        fu_new[mp] = 0
        u_new      = np.real(np.fft.ifft(fu_new))
        u          = u_new
        rhsp       = rhs

        # output to screen every 100 time steps
        if ((t+1)%1000==0):
            CFL = np.max(np.abs(u))*dt/dx          
            KE  = 0.5*np.var(u)
            out_t[save_t]   = (t+1)*dt 
            out_u[save_t,:] = u
            save_t += 1
            print("%d \t %f \t %f \t %f \t %f \t %f"%(t+1,(t+1)*dt,KE,CFL,np.max(u),np.min(u)))
        
        if((t+1)%1000==0):
            print(coeff)
            E1 = np.mean(-tau*dudx)
            E2 = np.mean(visc*dudx**2)
            E3 = np.mean(-tau*d3udx3)
            E4 = np.mean(dudx**3)
            E5 = np.mean(visc*d2udx2**2)
            print("%d \t %f \t %f \t %f \t %f \t %f"%(t+1,E1,E2,E3,E4,E5))

if __name__ == "__main__":
    main()