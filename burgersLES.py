#/usr/bin/env python
import pylab as pl
import numpy as np
from burgers import Utils, Settings, BurgersLES

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

    # if using Deardorff, initialize ke
    if model==4:
        kr = np.ones(nxLES)
    else:
        kr = 0

    # initialize random number generator
    np.random.seed(1)

    # place holder for right hand side
    rhsp = 0
 
    # advance in time
    for t in range(int(nt)):
        
        # compute derivatives
        derivs = utils.derivative(u,dx)
        dudx   = derivs['dudx']
        du2dx  = derivs['du2dx']
        d2udx2 = derivs['d2udx2']
        d3udx3 = derivs['d3udx3']

        # add fractional Brownian motion (FBM) noise
        f  = utils.noise(0.75,nxDNS)
        ff = utils.filterDown(f,int(nxDNS/nxLES))
        
        # compute subgrid terms
        sgs    = LES.subgrid(u,dudx,dx,kr)
        tau    = sgs["tau"]
        coeff  = sgs["coeff"]
        dtaudx = utils.derivative(tau,dx)["dudx"]
        if model==4:
            kr = sgs["kr"]

        # compute right hand side
        rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*ff - 0.5*dtaudx
        
        # advance in time
        if t == 0:
            # Euler
            u_new = u + dt*rhs
        else:
            # Adams-Bashforth 2nd
            u_new = u + dt*(1.5*rhs - 0.5*rhsp)
        
        fu_new     = np.fft.fft(u_new)
        fu_new[mp] = 0
        u_new      = np.real(np.fft.ifft(fu_new))
        u          = u_new
        rhsp       = rhs

        # output to screen every 100 time steps
        if ((t+1)%100==0):
            CFL = np.max(np.abs(u))*dt/dx          
            KE  = 0.5*np.var(u)
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