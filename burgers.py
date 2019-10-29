import numpy as np
import cmath as cm

class Burgers:

    # function to generate fractional Brownian motion (FBM) noise
    def addNoise(self,alpha,n):
        x     = np.sqrt(n)*np.random.randn(n)
        m     = int(n/2)
        k     = np.abs(np.fft.fftfreq(n,d=1/n))
        k[0]  = 1
        fx    = np.fft.fft(x)
        fx[0] = 0
        fx[m] = 0
        fx1   = fx * ( k**(-alpha/2) )
        x1    = np.real(np.fft.ifft(fx1) )
        return x1
    
    # function to compute spatial derivatives
    def computeDerivative(self,u,dx,opt=1):
        n = np.int(u.shape[0])
        m = int(n/2)
        
        # Fourier colocation method
        if opt==1:
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
            
            return [dudx,d2udx2,du2dx,d3udx3]