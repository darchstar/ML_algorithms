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
oldW = []

# Hyperparameter: like variation of parameter estimate and Acceleration data
# These hyperparameters will be annealed during MCMC optimization
sigmaMCMC = .01

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
for j in range(200):
    freq = ((100-40)*np.random.rand()+40)/60 # heart rate at random plausible freq
    freq2 = 12/60 # respiration rate
    phase = 2*np.pi*np.random.rand()*freq # phase component random | freq

    # Bernouli noise is like the event of a motion artifact; spurrious.
    #                This can also be seen as a kind of measurement noise
    #                Sign of this is random. can be negative or positive
    # Normal noise is another kind of noise that may be implicit in the sensor;
    # measurement

    normal = np.random.normal(0, .5, 1000)
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

    activation = np.random.rand()
    sinusAct = np.array([0 for i in range(300)] + [activation for i in range(400)] + [0 for i in range(300)])
    # Convolve with gaussian to smooth transition and multiply by sinusoid.
    # Assume sinusoid freq is 2Hz (within band of interested HR freqs). We should add the freq to the
    # parameters we're optimizing for in the posterior if it is also dynamic

    # Uncomment if you want to simulate a sinusoidal artifact
    sinusAct = 10*signal.sawtooth([2*np.pi*i*.5/100 for i in range(1000)])*sinusAct+ np.random.normal(0,1,1000)
    activityprior = np.array([0 for i in range(300)] + [1 for i in range(400)] + [0 for i in range(300)], dtype=float)
    activityprior *= signal.sawtooth([2*np.pi*i*.5/100 for i in range(1000)])
    # Components of pulse; sinusoid with harmonics with phase shifts. Looks like
    # Pulsatile Photoplethysmogram. Noise also observed.
    B = A[1]*np.sin([2*np.pi*i*freq*1/100 + phase for i in range(1000)])
    B += A[0]*np.sin([2*np.pi*i*freq2*1/100 for i in range(1000)])
    B += A[2]*np.sin([2*np.pi*i*2*freq*1/100 + 2*phase for i in range(1000)])
    B += A[3]*np.sin([2*np.pi*i*3*freq*1/100 + 4*phase for i in range(1000)])
    nonoise = B# + sinusAct
    nonoisesample.append(nonoise)
    noise = B + bernouliandnormal# + sinusAct
    noisedsamples.append(noise)

    # Initial state should be 0 or randomly initialized. I choose 0
    noisefix = np.hstack(([0 for i in range(20)], noise))

    # Fixing observation matrix for OLS AR Model
    observe = np.array([noisefix[i:i+20] for i in range(len(noisefix)-20)])
    nextobs = noisefix[20:]
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
    previousstate = noise[-20:]
    out = []
    variance = np.std(denoised - noise)

    for i in range(200):
        out.append(W.dot(previousstate))
        previousstate[:-1] = previousstate[1:]
        previousstate[-1] = out[-1] + np.random.normal(0,variance)
    #plt.plot([i*1/100 for i in range(1000)], out)
    #plt.show()

nobs = len(observs)
for j in range(1000):
    # MCMC Annealing part. W sampled from normal. sample <= 32 observs to
    # optimize posterior. Helps a lot with computational overhead. One hopes
    # such a q(W) converges to the real posterior. Use OLS estimate of first
    # observed PPG as intial prior estimate
    if j == 0:
        oldW = W
    else:
        pOld = 0 # Old posterior estimate
        pCandidate = 0 # Candidate posterior estimate
        # Sample only part of observations dist once sample size gets very large
        if nobs > 100:
            p = np.random.permutation(nobs)[:100]
        else:
            p = np.random.permutation(nobs)
        # Candidate to update posterior expectation

        for k in range(1):

            candidateW = oldW + np.random.normal(0,sigmaMCMC,len(oldW))
            for obs,y,ac in zip(np.array(observs)[p],np.array(labs)[p], np.array(accData)[p]):

            # If marginalizing activation out of likelihood, we should perform
            # an EM like algorithm. First optimize q(W|Artifact), update W, then
            # optimize q(W|Activation)

            # P(Y|X,W) = N(XW,sigma_a)
            # P(Artifact) = N(0,sigma_b)
            # P(Activation) = N(0, sigma_c)
            # P(B) = N(0, sigma_MCMC)
# P(W|Y,X,Artifact,Activation)  ~ P(Y|X,W)P(Artifact)P(Activation)P(B)
# P(W|Y,X,Artifact,Activation)  ~ 1/(2pisigma_a**2)*exp(-sum(y-XW)/(2sigma_a))*1/(2pisigma_b**2) * exp(-sum(artifact)/(2sigma_b)) *1/(2pisigma_c**2) * exp(-sum(activation)/(2sigma_c))

                # Evaluate Old Posterior
                oldexp = obs.dot(oldW.T)
                covobs = 2*(oldexp - oldexp.mean()).dot((y-y.mean()).T)
                covac = 2*(oldexp - oldexp.mean()).dot((ac - ac.mean()).T)
                covact = 2*(oldexp - oldexp.mean()).dot((activityprior -\
                    activityprior.mean()).T)

                if k == 0:
                    pOld += 1/(covobs/2*np.sqrt(2*np.pi))*np.exp(-np.sum((y - oldexp)**2)/covobs)* \
                            (1-1/(covac/2*np.sqrt(2*np.pi))*np.exp(-np.sum(ac**2)/covac))# * \
        #                    (1-1/(covact/2*np.sqrt(2*np.pi))*np.exp(-np.sum(activityprior**2)/covact))
                else:
                    pOld += 1/(covobs/2*np.sqrt(2*np.pi))*np.exp(-np.sum((y - oldexp)**2)/covobs)* \
                            (1-1/(covact/2*np.sqrt(2*np.pi))*np.exp(-np.sum(activityprior**2)/covact))
        #                    (1-1/(covac/2*np.sqrt(2*np.pi))*np.exp(-np.sum(ac**2)/covac))# * \

                # Evaluate Candidate
                candidateexp = obs.dot(candidateW.T)
                candcovobs = 2*(candidateexp - candidateexp.mean()).dot((y -\
                    y.mean()).T)
                candcovac = 2*(candidateexp - candidateexp.mean()).dot((ac - ac.mean()).T)
                candcovact = 2*(candidateexp -\
                        candidateexp.mean()).dot((activityprior - activityprior.mean()).T)

                if k == 0:
                    pCandidate += 1/(candcovobs/2*np.sqrt(2*np.pi))*np.exp(-np.sum((y - candidateexp)**2)/candcovobs - \
                        np.sum((candidateW - oldW)**2)/oldW.dot(candidateW.T)) *\
                        (1-1/(candcovac/2*np.sqrt(2*np.pi))*np.exp(-np.sum(ac**2)/candcovac))# * \
    #                (1-1/(candcovact/2*np.sqrt(2*np.pi))*np.exp(-np.sum(activityprior**2)/candcovact))
                else:
                    pCandidate += 1/(candcovobs/2*np.sqrt(2*np.pi))*np.exp(-np.sum((y - candidateexp)**2)/candcovobs - \
                        np.sum((candidateW - oldW)**2)/oldW.dot(candidateW.T)) *\
                        (1-1/(candcovact/2*np.sqrt(2*np.pi))*np.exp(-np.sum(activityprior**2)/candcovact))
    #                    (1-1/(candcovac/2*np.sqrt(2*np.pi))*np.exp(-np.sum(ac**2)/candcovac))# * \

                print(j, pCandidate/pOld, end='\r')

            # See if new posterior is more probable
            if pCandidate/pOld > 1:
                # Update posterior estimate
                oldW = candidateW
                variance = pCandidate
                print()
                print(j,"Changed!")
        # Annealing
        if j % 700== 0 and i != 0:
            sigmaMCMC /= 5

        Beta.append(oldW)

Beta = np.array(Beta)
Wsig = np.std(Beta[900:],0)

### Let's see how well we did.
time = [i*1/100 for i in range(1000)]
freqsfft = np.fft.fftfreq(1000, d=1/100)
for i in range(30):
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

    fig, ax = plt.subplots(4,1)
    ax = ax.ravel()
    ax[0].plot(time, nonoisesample[i], label="Real")
    ax[0].plot(time, res)
#    ax[0].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[0].plot(time, ores,alpha=.5, label="Bayes Filter")
    for m in marksL[i]:
        ax[0].axvline(x=m/100, color='g', alpha=.3)
    ax[0].set_ylabel("A.U")
    ax[1].plot(time, noisedsamples[i], label="Noised")
    ax[1].plot(time, res)
#    ax[1].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[1].plot(time, ores,alpha=.5, label="Bayes Filter")
    for m in marksL[i]:
        ax[1].axvline(x=m/100, color='g', alpha=.3)
    ax[2].plot(time, denoisedsamples[i], label="AR Denoised")
    ax[2].plot(time, res)
#    ax[2].fill_between(time, res+resstd, res-resstd, color="orange", alpha=.5)
    ax[2].plot(time, ores,alpha=.5, label="Bayes Filter")
    for m in marksL[i]:
        ax[2].axvline(x=m/100, color='g', alpha=.3)
    ax[2].set_xlabel("Time (seconds)")
    kalman = [denoisedsamples[i][len(oldW)-1]]
    for k in range(len(oldW), len(noisedsamples[i])):
        xk = denoisedsamples[i][k]
        covac = denoisedsamples[i][k-len(oldW):k].dot(accData[i][k-len(oldW):k])
        ac = accData[i][k-len(oldW):k]
        pArt = (1/(np.sqrt(abs(covac))*np.sqrt(2*np.pi))*np.exp(-np.sum(ac**2)/(2*abs(covac))))
        direction = np.sign(np.mean(ac))
        kalman.append(pArt*(xk-kalman[-1]) + kalman[-1])
    ax[3].plot(time[len(oldW)-1:], kalman, label="Kalman")
    ax[3].plot(time[len(oldW)-1:], nonoisesample[i][len(oldW)-1:], label="Real")
    ax[3].plot(time[len(oldW)-1:], ores[len(oldW)-1:], label="Bayes")

    for axx in ax:
        axx.legend(bbox_to_anchor=(1.1,1.05), fancybox=True, framealpha=0.5)

    fig.savefig("Generated_{0:d}.png".format(i), dpi=500)
    plt.close(fig)

