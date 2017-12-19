#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import warnings

'''
1:
See paper. Trick: Simplify with chain rule for grad of log summation exp.
pi(a|x) = exp(transpose(theta)phi(x,a))/exp(sum_{b for all A}{}{transpose(theta)*phi(x,b)}

Get:
grad(log(pi(a|x))) = phi(x,a) - sum_{b for all A}{}{transpose(theta)*phi(x,b)^2}

Goal: use gradient to update policy weights theta in iterations -> optimal policy weights
Like supervised learning

2:
    V starts off equal 0 for each state... only one state exists...
    Since V = p(0), it would lend that integral of grad(p) would give V

    2 actions exists, a1 and a2

    a1 gives reward with p1 = 1/2, a2 gives reward with p2 = 1/2 + delta, delta we choose

    R is bernoulli:
        1 w.p. pik
        0 otherwise
    with i being index of action

    theta is size R^1x2, given only two actions, 1 state
    instantiate random

    phi(x,b) is feature vector, function is a vector whose result is
        1 if b = a, ie if b is equal to the action taken
        0 otherwise
    phi is size R^1x2 as well

    figure out gk?
    Well, look at slide 38 of lecture:
        grad(p(theta_k))
        where
            grad(p(theta_k)) = sum_{for all x and a in states and actions}{
                            mu_0*pi_{theta}(a|x)*grad(log(pi(a|x)))*(Q^{pi_theta}(x,a) - h(x)
            where h(x)
                gamma*V^pi_k

    how do we solve Q_{k}(x,a)?
        Monte Carlo
            sample a1 several times for Q_{k}(x,a1)
            sample a2 several times for Q_{k}(x,a2)
                where Q is Expectations(rewards|x,a,pi)
                    or sum{i=0}{n}{gamma*R}
    h(x) = gamma*V(x)

    then plug into theta_{k+1} = theta_{k} + alpha_{k}*g_{k}
    gk being the grap(p_theta_k)

    iterate until converges, ie gk sufficiently small at threshold

2.1) Gamma is the discount factor and this affects how much the agent is likely
     to explore versus exploit rewards. Higher discounts implies more likely to
     explore than exploit.
'''

gamma = 0.99 # discount factor
k_array = [[] for i in range(4)]
d = [0.01, 0.05, 0.1, 0.5] # Various changes of a2's probability of rewards
warnings.filterwarnings('error') # So we can catch warnings
for n in range(4):
    np.random.seed(10) # reset random seed each trial
    theta = np.random.random((1,2)) # initialize random weight vector
    V = 0 # initial value is zero. Higher initial value may change how much we
          # explore
    its = 0
    maxIts = 2000 #maximum iterations
    Q = np.zeros((1,2)) # As well, Action Val function initialized zero
    alpha = .5 # initial step size
    delta = d[n] # the delta for this trial's a2 rewards
    gamma = .99
    phi = np.array([[1,0], [0,1]]) # Feature vector. If a=a1; phi[0] = [1, 0]
                                   # if a=a2, phi[1] = [0, 1]

    while its < maxIts:# and gk > threshold:
        its += 1
        # Fun play thing to see how changing rewards and feature
        # vector affects agent
        if its > 500:
            phi = np.array([[0,1],[1,0]])

        # Monte Carlo simulations
        # Action value is the expectations of discounted rewards
        # sum_{n=1}{N}{r_t + gamma*V_pi}
        for i in range(2):
            for j in range(1000):
                if i == 0:
                    if its < 500:
                        if np.random.rand() > .50:
                            Q[0,i] += 1 + gamma*V
                        else:
                            Q[0,i] += gamma*V
                    elif its > 500:
                        if np.random.rand() > .50 - delta:
                            Q[0,i] += 1 + gamma*V
                        else:
                            Q[0,i] += gamma*V
                elif i == 1:
                    if its < 500:
                        if np.random.rand() > .50 - delta:
                            Q[0,i] += 1 + gamma*V
                        else:
                            Q[0,i] += gamma*V
                    elif its > 500:
                        if np.random.rand() > .50:
                            Q[0,i] += 1 + gamma*V
                        else:
                            Q[0,i] += gamma*V
            Q[0,i] /=1000

        dlogpi = phi - theta.dot(np.sum([p.dot(p.T) for p in phi]).T) # derived
        try:
            pi = np.exp(theta.dot(phi.T))
            pi /= np.exp(theta.dot(np.sum(phi,1).T))
            h = gamma*V
            g = np.sum(1*pi[0]*dlogpi*(Q - h))
            # Play with step size here to see how it affects convergence
            # I prefer to decay the steps by its^2 because of the quick
            # convergence and stable results, though it tends to underestimate
            # optimal policy pi
            theta = theta + alpha/(its**2) * g
        except:
            # This tells us if solution diverged
            print("Runtime Warning n = " + str(n))


        V += gamma*g # as I mentionned before, we can call our Value
                     # the sum of the grad(p(theta)) which is g
        k_array[n].append(pi[0])

k_array = np.array(k_array)
k = [j for j in range(maxIts)]
for i in range(4):
    plt.subplot(211)
    plt.title("a2")
    plt.plot(k, k_array[i,:,1])
    plt.subplot(212)
    plt.title("a1")
    plt.plot(k,k_array[i,:,0])
plt.legend([".01", "0.05", "0.1", "0.5"])
plt.show()
