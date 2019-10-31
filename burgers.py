import numpy as np
import cmath as cm

class Utils:

    # function to generate fractional Brownian motion (FBM) noise
    def noise(self,alpha,n):
        x     = np.sqrt(n)*np.random.randn(n)
        m     = int(n/2)
        k     = np.abs(np.fft.fftfreq(n,d=1/n))
        k[0]  = 1
        fx    = np.fft.fft(x)
        fx[0] = 0
        fx[m] = 0
        fx1   = fx * ( k**(-alpha/2) )
        x1    = np.real(np.fft.ifft(fx1))
        
        return x1
    
    # function to compute spatial derivatives
    def derivative(self,u,dx):
        
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
    
    # Fourier filtering
    def filter(self,u,k):
            
        n   = int(u.shape[0])
        m   = int(n/k)
        l   = int(m/2)
        
        fu  = np.fft.fft(u)
        fuf = np.zeros(m,dtype=np.complex)
        
        fuf[0:l]   = fu[0:l]
        fuf[l+1:m] = fu[n-l+1:n]
        
        uf = (1/k)*np.real(np.fft.ifft(fuf))

        return uf