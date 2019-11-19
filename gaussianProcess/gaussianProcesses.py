import numpy as np
import numba as nb

class GP():
    def __init__(self):
        return

    # Density Estimates
    # Fast with numba
    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def gaussianKernelDensity(data, sigma=1):
        p = np.zeros(len(data), dtype=np.float64)
        # Subset estimation with select.
#        select = np.random.permutation(len(data))[:1000]
        for d in range(data):
            p += 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(data-d)**2/(2*sigma**2))
        return p/data.size

    @staticmethod 
    @nb.njit(fastmath=True,parallel=True)
    def gaussian2dKernelDensity(data, sigma=np.array([[1,0],[0,1]],dtype=np.float64)):
        useData = data.T
        p = np.zeros(len(useData), dtype=np.float64)
        for i in range(len(useData)):
            for j in range(len(useData)):
                p[i] += 1/np.sqrt((2*np.pi)**2*np.linalg.det(sigma))*\
                        np.exp(-((useData[j,:] - useData[i,:]).T\
                                    .dot(np.linalg.pinv(sigma))\
                                    .dot(useData[j,:] - useData[i,:])))
        return p/len(useData)
    
    
    # Kernels
    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def maternKernel(a, b, ld=3):
        cmat = np.zeros((len(b), len(a)),dtype=np.float64)
        rationing = np.array([ld for i in range(a.shape[1])], dtype=np.float64)
        for i in range(len(b)):
            r = np.sqrt(np.sum((a - b[i])**2/rationing,1))
            cmat[i,:] = (1 + np.sqrt(5) * r + 5*r**2/(3))
            cmat[i,:] = cmat[i,:] * np.exp(-np.sqrt(5)*r)
        return cmat

    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    # derivative maternKernel
    def dmaternKernel(a, b, ld=3):
        dcmat = np.zeros((len(b), len(a)),dtype=np.float64)
        rationing = np.array([ld for i in range(a.shape[1])], dtype=np.float64)
        for i in range(len(b)):
            r = np.sqrt(np.sum((a - b[i])**2,1))
            dcmat[i,:] = (-np.sqrt(5)*r/ld**2 -2*5*r**2/ld**3)*np.exp(-np.sqrt(5)*r/ld)
            dcmat[i,:] = dcmat[i,:] + (1+ np.sqrt(5)*r/ld +\
                    5*r**2/ld**2)*np.exp(-np.sqrt(5)*r/ld)*np.sqrt(5)*r/ld**2
        return dcmat

    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def maternSinKernel(a, b, ld=3):
        cmat = np.zeros((len(b), len(a)),dtype=np.float64)
        for i in range(len(b)):
            r = np.sin(np.sqrt(np.sum((a - b[i])**2,1))*np.pi)
            cmat[i,:] = (1 + np.sqrt(5) * r/ld + 5*r**2/(3*ld**2))
            cmat[i,:] = cmat[i,:] * np.exp(-np.sqrt(5)*r/ld)
        return cmat
    
    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def squaredExpSinKernel(a, b, ld=3):
        cmat = np.zeros((len(b), len(a)),dtype=np.float64)
        for i in range(len(b)):
            r = np.sin(np.sqrt(np.sum((a - b[i])**2,1))*np.pi)
            cmat[i,:] = np.exp(-r**2/ld)
        return cmat
    
    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def squaredExpKernel(a, b, ld=3):
        cmat = np.zeros((len(b), len(a)),dtype=np.float64)
        rationing = np.array([ld for i in range(a.shape[1])], dtype=np.float64)
        for i in range(len(b)):
            r = np.sqrt(np.sum((a - b[i])**2/rationing,1))
            cmat[i,:] = np.exp(-r**2)
        return cmat

    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    # derivative squared exponential
    def dSquareExpKernel(a, b, ld=3):
        cmat = np.zeros((len(b), len(a)), dtype=np.float64)
        rationing = np.array([ld for i in range(a.shape[1])], dtype=np.float64)
        for i in range(len(b)):
            r = np.sqrt(np.sum((a - b[i])**2/rationing,1))
            cmat[i,:] = -1/(2*ld)*np.exp(-r**2/(2*ld))*1/(2*r)
        return cmat
    
    def WienerKernel(self,a, b, ld=3.):
        cmat = np.zeros((len(b), len(a)), dtype=np.float64)
        for i in range(len(b)):
            tempb = np.array(np.sqrt(np.sum(b[i]**2)))
            tempa = np.sqrt(np.sum(a**2,1))
            cmat[i,:] = np.min((tempb.repeat(len(tempa)),tempa),0)
        return cmat/cmat.max()

    def optimizeHyperparms(self, data, inp, lr=.001, ld=1., sigma=1., noise=.1, niter=100):
        data = data-data.mean(0)
        inp = inp - inp.mean(0)
        sigma = np.array(sigma)
        noise = np.array(noise)
        # Gradient Ascent: Marginal maximum Log Likelihood p(y|hyperparameters)
        for i in range(niter):
            print(ld, sigma, noise)
            K = sigma.squeeze()*self.maternKernel(inp, inp, ld) + noise*np.eye(len(data))
            Kinv = np.linalg.pinv(K)
            dK = sigma.squeeze()*self.dmaternKernel(inp,inp, ld)
            dpdl = (.5*data.T.dot(Kinv).dot(dK).dot(Kinv).dot(data) - .5*np.diagonal(Kinv.dot(dK)).sum())
            dpds = (.5*data.T.dot(Kinv).dot(K/sigma).dot(Kinv).dot(data) - .5*np.diagonal(Kinv.dot(K/sigma)).sum())
            dpdn = (.5*data.T.dot(Kinv).dot(np.eye(len(Kinv))).dot(Kinv).dot(data) - .5*np.diagonal(Kinv.dot(np.eye(len(Kinv)))).sum())
            ld = (ld + lr*dpdl).squeeze()
            sigma = (sigma + lr*dpds).squeeze()
            noise = (noise + lr*dpdn).squeeze()
            # Constraints: Must be > 0
            ld = abs(ld) + 1e-10
            sigma = abs(sigma) + 1e-10
            noise = abs(noise) + 1e-10
        self.ld = ld
        self.sigma = sigma
        self.noise = np.sqrt(noise)
        return ld, sigma, np.sqrt(noise)


    # Posterior Inference
    def posteriorReg(self, inp, observations, test, ld=1, sig=1, target="matern", noise=0):
        # normalizing 
        inp = (inp - inp.mean(0))
        test = (test - test.mean(0))
        obsmean = observations.mean(0)
        observations = (observations - observations.mean(0))

        # Check input dimension. Should be same for test and input
        if inp.shape[1] > test.shape[1]:
            test = np.vstack([test.T, np.array([np.zeros(len(test)) for i in range(inp.shape[1] - test.shape[1])])]).T
        elif test.shape[1] > inp.shape[1]:
            inp = np.vstack([inp.T, np.array([np.zeros(len(inp)) for i in range(test.shape[1] - inp.shape[1])])]).T

        # Choose kernel
        if target == "sin":
            kernel = lambda t, i, l: self.maternSinKernel(t,i,l) + self.maternKernel(t,i,l)
        elif target == "squaredexp":
            kernel = self.squaredExpKernel
        elif target == "ns_squaredexp":
            kernel = lambda t, i, l: self.squaredExpKernel(t,i,l) + self.WienerKernel(t,i,l)
        elif target == "matern":
            kernel = self.maternKernel
        elif target == "ns_matern":
            kernel = lambda t, i, l: self.maternKernel(t,i,l) + self.WienerKernel(t,i,l)
        else:
            kernel = self.maternKernel

        # Test kernel
        tk = sig*kernel(test, test, ld)

        # Train kernel
        inpk = sig*kernel(inp, inp,ld)

        # Test given train kernel
        inptk = sig*kernel(test, inp,ld)
        invinpk = np.linalg.pinv(inpk + np.eye(len(inpk))*noise**2) 
        # Covariance update
        posteriorK = tk - inptk.T.dot(invinpk).dot(inptk)
        
        # Predicted test vals
        posteriormu = inptk.T.dot(invinpk).dot(observations)
        posteriormu = (posteriormu.T + obsmean).T
        
        # Regression kernel
        regMat = inptk.T.dot(invinpk)

        return posteriormu, posteriorK
