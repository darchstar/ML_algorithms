import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy import signal

def gaussian(inp, sigma=1):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(inp)**2/(2*sigma**2))

noisedsamples = []
nonoisesample = []
denoisedsamples = []
accData = []
observs = []
labs = []
Beta = []
marksL = []
old = 0
b,a = signal.butter(3, 15/50, 'lowpass')

# Hyperparameters: like variation of parameter estimate and Acceleration data
# These hyperparameters will be annealed during MCMC optimization
sigmaMCMC = 0.01
sigmawhite = 0.1
sigmaAc = 1
sigmasinusAct = 1

# Bayesian Learning
# Sample data | noise, i.e. Y|Artifact,W
# Sample W|sigma_w as well.
# Optimize Posterior W | Y, Artifact, sigma_w
# Annealing procedure, i.e. MCMC
# In theory we could parameterize acceleration into our model into, but I only
# use it to optimize posterior to marginalize it out of our likelihood of
# observation given our autoregressive parameters. Makes acceleration data
# unnecessary

# Possible improvements: transforming regression model features using kernel.
# RBF can be considered, for example. Or any other nonlinear feature transform.

# Sample observations
for j in range(100):
    freq = ((100-40)*np.random.rand()+40)/60 # heart rate at random plausible freq
    freq2 = 12/60 # respiration rate
    phase = 2*np.pi*np.random.rand()*freq # phase component random | freq

    # Bernouli noise is like the event of a motion artifact; spurrious.
    #                This can also be seen as a kind of measurement noise
    #                Sign of this is random. can be negative or positive
    # Normal noise is another kind of noise that may be implicit in the sensor;
    # measurement

    normal = np.random.normal(0, 2, 1000)
    bernouli = 5*np.sign(np.random.rand(1000)-.5)*np.random.binomial(1, 1/100, 1000)/1
    marks = np.where(abs(bernouli) > 0)
    marksL.append(marks[0])

    # convolve with gaussian so that acceleration artifacts have spread
    gauss = gaussian(np.array([i*.125 for i in range(-24,24)]))
    convolvedbern = np.convolve(bernouli, gauss,'same')
    bernouliandnormal = normal + convolvedbern
    # Simulate acceleration sensor data to condition posterior on this too,
    # marginalize this out of the likelihood of our observation
    acceleration = np.random.normal(0,.1,1000) + convolvedbern
    accData.append(acceleration)

    A = [(1-.5)*np.random.rand() + .5, 1, .4, .2] # amplitude at each harmonic

    # Uncomment if you want to simulate a sinusoidal artifact
    activation = np.random.rand()
    sinusAct = np.array([0 for i in range(300)] + [activation for i in range(400)] + [0 for i in range(300)])
    # Convolve with gaussian to smooth transition and multiply by sinusoid.
    # Assume sinusoid freq is 2Hz (within band of interested HR freqs). We should add the freq to the
    # parameters we're optimizing for in the posterior if it is also dynamic
    sinusAct = 10*np.sin([2*np.pi*i*2/100 for i in range(1000)])*np.convolve(sinusAct, gauss, 'same')/10
    activityprior = np.array([0 for i in range(300)] + [1 for i in range(400)] + [0 for i in range(300)], dtype=float)
    activityprior *= 10*np.sin([2*np.pi*i*2/100 for i in range(1000)])
    # Components of pulse; sinusoid with harmonics with phase shifts. Looks like
    # Pulsatile Photoplethysmogram. Noise also observed.
    B = A[1]*np.sin([2*np.pi*i*freq*1/100 + phase for i in range(1000)])
    B += A[0]*np.sin([2*np.pi*i*freq2*1/100 for i in range(1000)])
    B += A[2]*np.sin([2*np.pi*i*2*freq*1/100 + 2*phase for i in range(1000)])
    B += A[3]*np.sin([2*np.pi*i*3*freq*1/100 + 4*phase for i in range(1000)])
    nonoise = B + sinusAct
    nonoisesample.append(nonoise)
    noise = B+ bernouliandnormal + sinusAct
    noisedsamples.append(noise)

    # Initial state should be 0 or randomly initialized. I choose 0
    noisefix = np.hstack(([0 for i in range(40)], noise))

    # Fixing observation matrix for OLS AR Model
    observe = np.array([noisefix[i:i+40] for i in range(len(noisefix)-40)])
    nextobs = noisefix[40:]
    observs.append(observe)
    labs.append(nextobs)

    # Ordinary least squares obtained by minimizing (Y - WX)^2
    # Solution is where dL/dW = 0
    # Y^2 - 2WXY + (WX)^2 = L
    # 2WXX^T - 2XY = 0
    # W = (XX^T)^-1XY
    W = linalg.pinv(np.dot(observe.T, observe)).dot(observe.T).dot(nextobs)

    denoised = observe.dot(W.T)
    denoisedsamples.append(denoised)


    ### See how well the autoregressive model can predict next points ###
    previousstate = noise[-40:]
    out = []
    variance = np.std(denoised - noise)

    for i in range(1000):
        out.append(W.dot(previousstate))
        previousstate[:-1] = previousstate[1:]
        previousstate[-1] = out[-1] + np.random.normal(0,variance)
    #plt.plot([i*1/100 for i in range(1000)], out)
    #plt.show()

nobs = len(observs)
for j in range(2000):
    # MCMC Annealing part. W sampled from normal. sample <= 32 observs to
    # optimize posterior. Helps a lot with computational overhead. One hopes
    # such a q(W) converges to the real posterior. Use OLS estimate of first
    # observed PPG as intial prior estimate
    if j == 0:
        oldW = W
    else:
        var2 = 0 # Old posterior estimate
        var3 = 0 # Candidate posterior estimate
        # Sample only part of observations dist once sample size gets very large
        if nobs > 32:
            p = np.random.permutation(nobs)[:32]
        else:
            p = np.random.permutation(nobs)
        # Candidate to update posterior expectation
        candidateW = oldW + np.random.normal(0,sigmaMCMC,len(oldW))
        for obs,noi,ac in zip(np.array(observs)[p],np.array(labs)[p], np.array(accData)[p]):

            # If marginalizing activation out of likelihood, we should perform
            # an EM like algorithm. First optimize q(W|Artifact), update W, then
            # optimize q(W|Activation)

            for k in range(2):
                # Noise added to error to avoid overfitting
                oldexp = obs.dot(oldW.T)
                var2 += np.std((oldexp - noi + np.random.normal(0,sigmawhite,1000)))
                standold = oldexp#(oldexp - oldexp.mean())/oldexp.std()
                if k == 0:
                    var2 += np.std((oldexp - sigmaAc*ac))
                else:
                    var2 += np.std((oldexp - sigmasinusAct*activityprior))
                candidateexp = obs.dot(candidateW.T)
                var3 += np.std((candidateexp - noi + np.random.normal(0,sigmawhite,1000)))
                if k == 0:
                    var3 += np.std((candidateexp - sigmaAc*ac))
                else:
                    var3 += np.std((candidateexp - sigmasinusAct*activityprior))
                print(j, var3/var2, end='\r')
        if 1 > var3/var2:
            # Update posterior estimate
            oldW = candidateW
            variance = var3
            print(j,"Changed!")
        # Annealing
        if j % 200== 0 and i != 0:
            sigmaMCMC /= 10
            sigmawhite /= 2
        if j % 300 == 0 and i != 0:
            sigmaAc /= 2
            sigmasinusAct /= 2

        Beta.append(oldW)

Beta = np.array(Beta)
Wsig = np.std(Beta[900:],0)

### Let's see how well we did.
time = [i*1/100 for i in range(1000)]
freqsfft = np.fft.fftfreq(1000, d=1/100)
for i in range(len(nonoisesample)-100,len(nonoisesample)):
    var = np.std(observs[i].dot(oldW.T) - noisedsamples[i])
    oreslist = []
    reslist = []
    #### Finding Confidence Intervals ###
#    for j in range(50):
#        tempW = np.zeros(*oldW.shape)
#        for k in range(40):
#            tempW[i] = oldW[i] + np.random.normal(0,Wsig[i])
#        oreslist.append(observs[i].dot(tempW.T))
#        reslist.append(signal.filtfilt(b,a,observs[i].dot(tempW.T)))
#    ores = np.mean(oreslist, 0)
#    res = np.mean(reslist,0)
#    resstd = np.std(reslist,0)
#    oresstd = np.std(oreslist,0)

    ores = observs[i].dot(oldW.T)
    res = signal.filtfilt(b,a,ores)

    fig, ax = plt.subplots(3,1)
    ax = ax.ravel()
    ax[0].plot(time, nonoisesample[i])
    ax[0].plot(time, res)
#    ax[0].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[0].plot(time, ores,alpha=.5)
    for m in marksL[i]:
        ax[0].axvline(x=m/100, color='g', alpha=.3)
    ax[0].set_ylabel("A.U")
    ax[0].legend(["Real + Nuisance Sinusoidal Activity", "Bayes Filtering"])
    ax[1].plot(time, noisedsamples[i])
    ax[1].plot(time, res)
#    ax[1].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[1].plot(time, ores,alpha=.5)
    for m in marksL[i]:
        ax[1].axvline(x=m/100, color='g', alpha=.3)
    ax[1].legend(["Noised", "Bayes Filtering"])
    ax[2].plot(time, denoisedsamples[i])
    ax[2].plot(time, res)
#    ax[2].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[2].plot(time, ores,alpha=.5)
    for m in marksL[i]:
        ax[2].axvline(x=m/100, color='g', alpha=.3)
    ax[2].legend(["AR Denoised", "Bayes Filtering"])
    ax[2].set_xlabel("Time (seconds)")
    plt.figure()

    plt.plot(freqsfft, abs(np.fft.fft(nonoisesample[i])))
    plt.plot(freqsfft, abs(np.fft.fft(res)), '--')
    plt.figure()

    initialstate = res[-40:] + np.random.normal(0,var,40)
    out = []
    for j in range(1000):
        out.append(initialstate.dot(oldW.T))
        initialstate[:-1] = initialstate[1:]
        initialstate[-1] = out[-1] + np.random.normal(0,var)
    plt.plot(time, out)
    fig.savefig("Generated_{0:d}.png".format(i), dpi=1200)
    plt.close(fig)
