import json
import numpy as np
import cmath as cm
import netCDF4 as nc
from scipy.stats import norm

# class with helper utilities
class Utils:

    # function to generate fractional Brownian motion (FBM) noise
    def noise(self,alpha,n):
        x     = np.sqrt(n)*norm.ppf(np.random.rand(n))
        m     = int(n/2)
        k     = np.abs(np.fft.fftfreq(n,d=1/n))
        k[0]  = 1
        fx    = np.fft.fft(x)
        fx[0] = 0
        fx[m] = 0
        fx1   = fx * ( k**(-alpha/2) )
        x1    = np.real(np.fft.ifft(fx1))
        
        return x1
    
    # function to compute spatial derivatives in spectral space
    def derivative(self,u,dx):
        
        # signal shape information
        n = int(u.shape[0])
        m = int(n/2)
        
        # Fourier colocation method
        h       = 2*np.pi/n
        fac     = h/dx
        k       = np.fft.fftfreq(n,d=1/n)
        k[m]    = 0
        fu      = np.fft.fft(u)
        dudx    = fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu))
        d2udx2  = fac**2 * np.real(np.fft.ifft(-k*k*fu))
        d3udx3  = fac**3 * np.real(np.fft.ifft(-cm.sqrt(-1)*k**3*fu))
        
        # dealiasing needed for du2dx using zero-padding 
        zeroPad = np.zeros(n)
        fu_p    = np.insert(fu,m,zeroPad)
        u_p     = np.real(np.fft.ifft(fu_p))        
        u2_p    = u_p**2
        fu2_p   = np.fft.fft(u2_p)
        fu2     = fu2_p[0:m]
        fu2     = np.append(fu2,fu2_p[n+m:])
        du2dx   = 2*fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu2))

        # store derivatives in a dictionary for selective access
        derivatives = {
            'dudx'  :   dudx,
            'du2dx' :   du2dx,
            'd2udx2':   d2udx2,
            'd3udx3':   d3udx3
        }
        
        return derivatives
    
    # Fourier filtering from DNS to LES
    def filterDown(self,u,k):
        
        # signal shape information
        n   = int(u.shape[0])
        m   = int(n/k)
        l   = int(m/2)
        
        # compute fft then filter
        fu  = np.fft.fft(u)
        fuf = np.zeros(m,dtype=np.complex)
        fuf[0:l]   = fu[0:l]
        fuf[l+1:m] = fu[n-l+1:n]
        
        # return from spectral space
        uf = (1/k)*np.real(np.fft.ifft(fuf))

        return uf

    # Fourier box filter
    def filterBox(self,u,k):
        
        # signal size information
        n   = int(u.shape[0])
        m   = int(n/k)
        l   = int(m/2)
        
        # compute fft then filter
        fu  = np.fft.fft(u)
        fuf = np.zeros(n,dtype=np.complex)
        fuf[0:l]     = fu[0:l]
        fuf[n-l+1:n] = fu[n-l+1:n]

        # return from spectral space
        uf = np.real(np.fft.ifft(fuf))

        return uf
    
    # function to de-alias
    def dealias1(self,x,n):
        
        # signal size information
        m   = int(n/2)
        
        # compute fft then de-alias
        fx  = np.fft.fft(x)
        fxp = np.concatenate((fx[0:m+1],np.zeros(m),fx[m+1:n]))
        
        # return from spectral space
        xp  = np.real(np.fft.ifft(fxp))
        
        return xp
    
    # function to de-alias
    def dealias2(self,xp,n):
        
        # signal size information
        m     = int(n/2)
        
        # compute fft then de-alias
        fxp   = np.fft.fft(xp)
        fx    = np.concatenate((fxp[0:m+1],fxp[2*m+1:m+n]))
        fx[m] = 0
        
        # return from spectral space
        x     = (3/2)*np.real(np.fft.ifft(fx))
        
        return x

# class to read input settings
class Settings:

    # initializer to get settings from provided namelist
    def __init__(self,namelist):
        with open(namelist) as json_file:
            data = json.load(json_file)
        self.nxDNS = data["dns"]["nx"]
        self.nxLES = data["les"]["nx"]
        self.sgs   = data["les"]["sgs"]
        self.nt    = data["nt"]
        self.dt    = data["dt"]
        self.visc  = data["visc"]
        self.damp  = data["damp"]

# class to model subgrid terms
class BurgersLES:

    # initializer to get selected subgrid model
    def __init__(self,model):
        self.model = model
        if self.model==0:
            print("[pyBurgers: SGS] \t Running with no model")
        if self.model==1:
            print("[pyBurgers: SGS] \t Constant-coefficient Smagorinsky")
        if self.model==2:
            print("[pyBurgers: SGS] \t Dynamic Smagorinsky")
        if self.model==3:
            print("[pyBurgers: SGS] \t Dynamic Wong-Lilly")
        if self.model==4:
            print("[pyBurgers: SGS] \t Deardorff 1.5-order TKE")
    
    # function to compute subgrid terms
    def subgrid(self,u,dudx,dx,kr):
        
        # signal size information
        n = int(u.shape[0])

        # instantiate helper classes
        utils    = Utils()
        settings = Settings('namelist.json')
        
        # no model
        if self.model==0:
            sgs = {
                'tau'   :   np.zeros(n),
                'coeff' :   0
            }
            return sgs
        
        # constant coefficient Smagorinsky
        if self.model==1:
            CS2   = 0.16**2
            d1    = utils.dealias1(np.abs(dudx),n)
            d2    = utils.dealias1(dudx,n)
            d3    = utils.dealias2(d1*d2,n)
            tau   = -2*CS2*(dx**2)*d3
            coeff = np.sqrt(CS2)

            sgs = {
            'tau'   :   tau,
            'coeff' :   coeff
            }
            return sgs
        
        # dynamic Smagorinsky
        if self.model==2:
            uf    = utils.filterBox(u,2)
            uuf   = utils.filterBox(u**2,2)
            L11   = uuf - uf*uf
            dudxf = utils.filterBox(dudx,2)
            T     = np.abs(dudx)*dudx
            Tf    = utils.filterBox(T,2)   
            M11   = -2*(dx**2)*(4*np.abs(dudxf)*dudxf - Tf )
            if np.mean(M11*M11) == 0:
                CS2 = 0
            else:
                CS2 = np.mean(L11*M11)/np.mean(M11*M11)
            if CS2 < 0: 
                CS2 = 0
            d1    = utils.dealias1(np.abs(dudx),n)
            d2    = utils.dealias1(dudx,n)
            d3    = utils.dealias2(d1*d2,n)
            tau   = -2*CS2*(dx**2)*d3
            coeff = np.sqrt(CS2)
            
            sgs = {
                'tau'   :   tau,
                'coeff' :   coeff
            }
            return sgs
        
        # dynamic Wong-Lilly
        if self.model==3:
            uf    = utils.filterBox(u,2)
            uuf   = utils.filterBox(u**2,2)
            L11   = uuf - uf*uf
            dudxf = utils.filterBox(dudx,2)
            M11   = 2*(dx**(4/3))*dudxf*(1-2**(4/3))
            if np.mean(M11*M11) == 0:
                CWL = 0
            else:
                CWL = np.mean(L11*M11)/np.mean(M11*M11)
            if CWL < 0:
                CWL = 0
            d1    = utils.dealias1(np.abs(dudx),n)
            d2    = utils.dealias1(dudx,n)
            d3    = utils.dealias2(d1*d2,n)
            tau   = -2*CWL*(dx**(4/3))*dudx
            coeff = CWL

            sgs = {
                'tau'   :   tau,
                'coeff' :   coeff
            }
            return sgs
        
        # Deardorff TKE
        if self.model==4:
            Ce = 0.70
            C1 = 0.1
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

        # exception when none selected
        else:
            raise Exception("Please choose an SGS model in namelist.\n\
            0=no model\n\
            1=constant-coefficient Smagorinsky\n\
            2=dynamic Smagorinsky\n\
            3=dynamic Wong-Lilly\n\
            4=Deardorff 1.5-order TKE")