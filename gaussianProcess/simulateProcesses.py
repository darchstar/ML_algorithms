import numpy as np

# Some nice complex signals to simulate
def lorenzGenerator(initial_state=[1,1,1], N=10000, sigma=10, beta=8/3, rho=28, noise=1e-1, h=1e-2):
    x = np.zeros(N,dtype=float)
    y = np.zeros(N,dtype=float)
    z = np.zeros(N,dtype=float)
    initial_state = np.array(initial_state)
    x[0] = initial_state[0]
    y[0] = initial_state[1]
    z[0] = initial_state[2]

    def dxdydz(xt, yt, zt):
        dx = sigma*(yt - xt)
        dy = xt*(rho - zt) - yt
        dz = xt*yt - beta*zt
        return np.array([dx, dy, dz])

    for i, (xt,yt,zt) in enumerate(zip(x[:-1], y[:-1], z[:-1])):
        # Runge Kutta integration
        k1 = h*dxdydz(xt,yt,zt)
        ink2 = (xt + k1[0]/2, yt + k1[1]/2, zt + k1[2]/2,)
        k2 = h*dxdydz(*ink2)
        ink3 = (xt + k2[0]/2, yt + k2[1]/2, zt + k2[2]/2,)
        k3 = h*dxdydz(*ink3)
        ink4 = (xt + k3[0], yt + k3[1], zt + k3[2],)
        k4 = h*dxdydz(*ink4)
        x[i+1] = xt + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + np.random.randn()*noise
        y[i+1] = yt + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + np.random.randn()*noise
        z[i+1] = zt + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + np.random.randn()*noise
    return x,y,z

def henonMapGenerator(initial_state=[1,1], N=10000, alpha=1.4, beta=.3, noise=1e-1):
    x = np.zeros(N,dtype=float)
    y = np.zeros(N,dtype=float)
    x[0] = initial_state[0]
    y[0] = initial_state[1]
    for i, (xt,yt) in enumerate(zip(x[:-1], y[:-1])):
        x[i+1] = 1- alpha*xt**2 + yt + np.random.randn()*noise
        y[i+1] = beta*xt + np.random.randn()*noise
    return x,y

def generateCorrelatedNoise(cov=[1, .5, .3], N=10000):
    k = len(cov)
    cov = np.vstack([np.roll(cov,n) for n in range(k)])
    rM = np.random.randn(k,N)
    return np.linalg.cholesky(cov).dot(rM)
