import torch
from torch import nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
import matplotlib
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class rGenerator(nn.Module):
    def __init__(self, nF=1):
        super(rGenerator, self).__init__()

        self.genT = nn.GRU(nF + 1, 1, 1, batch_first=True)
        self.nFeatures = nF

    def forward(self, size, batch, nC):
        # conditioning labels
        labels = torch.randint(nC, (batch,1)).repeat(1,size).float()
        hidden = None
        xi = torch.rand(batch, size, self.nFeatures+1)
        xi[:,:,-1] = labels
        out, hidden = self.genT(xi, hidden)

        # tanh to give output a nonlinearity mapping; normalize output from
        # [-1,1]. Gives nice function for backpropagating loss to generator
        # layers. Also, steeper gradients than sigmoid is also a nice
        # characteristic
        return out.tanh(), labels

class rDiscriminator(nn.Module):
    def __init__(self, nF=1):
        super(rDiscriminator, self).__init__()

        self.discT = nn.GRU(nF+1,1,1, batch_first=True)

    def forward(self, data):
        hidden = None
        out, hidden = self.discT(data, hidden)

        return out

if __name__ == "__main__":
    # Hyperparms
    batch = 32
    epochs = 1000
    size = 600
    lr = 1e-3
    clipval = .01

    discriminator = rDiscriminator().cuda()
    generator = rGenerator().cuda()

    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

    ### Put data loading code here ###

    # Shuffle Data #
    p = np.random.permutation(len(data))
    data = data[p]
    for i in range(epochs):
        for j in range(0, data.size(0), batch):

            # Discriminator Stage #
            for k in range(5): # Train more because we want to have a good
                               # critic for the generator
                optimizerD.zero_grad()

                validres = discriminator(data[j:j+batch])

                genD, labs = generator(size, len(data[j:j+batch]))
                faked = torch.cat((genD, labs.unsqueeze(2)), dim=2)
                fakeres = discriminator(faked.detach())

                # Loss for how well the discrminator can detect fakes
                ### Cross Entropy: -E(t0log(p0) + t1log(p1)). E: expectation
                ###                                           t_i: class label
                ###                                           p_i: probability
                ### Since we want the discriminator to detect fakes, and t0
                ### corresponds to the true class, replace p1 with 1-p1. Also, let's
                ### perform label smoothing so that the discriminator doesn't get
                ### confident.
                # d_loss = -torch.mean(.9*torch.log(validres.sigmoid()) + torch.log(1-fakeres.sigmoid()))


                ### Wasserstein Loss: -(E[D(x)] - E[D(G_z)]) G(z): Generator output
                ###                                          D(x): Discriminator output
                ###                                          z: latent features
                ###                                          x: training batch
                ### We're trying to maximize the distance between the distance of
                ### the (i.e. find supremum) of the distance between our two
                ### distributions with our discriminator. The negative turns the
                ### maximization problem to a minimization one. Clipping the
                ### gradient is done to enforce the Lipschitz constraint
                d_loss = -(torch.mean(validres) - torch.mean(fakeres))
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clipVal)
                optimizerD.step()

            # Generator Stage #
            optimizerG.zero_grad()

            genD, labs = generator(size, len(data[j:j+batch]))
            faked = torch.cat((genD, labs.unsqueeze(2)), dim=2)
            fakeres = discriminator(faked)
            # Loss for how well we fool the discriminator

            ### Cross Entropy: -E(t0log(p0) + t1log(p1)) E: expectation
            ###                                           t_i: class label
            ###                                           p_i: probability
            ### Since we wanna fool the discriminator, and t0 corresponds to the
            ### true class, t0 =1; t1 = 0
            #lossG = -torch.mean(torch.log(fakeres.sigmoid()))

            ### Wasserstein Loss: -E[D(G_z)]. G(z): Generator output
            ###                               D(f): Discriminator output
            ###                               z: latent features
            ### We want to make sure that the Wasserstein distance does not get
            ### too large, so we force the generator loss to counter the
            ### discriminator's loss

            lossG = -torch.mean(fakeres)

            lossG.backward()
            optimizerG.step()

            print("Epoch {0:d}, G Loss {1:.3f}, D Loss {2:.3f}".format(i, lossG.item()/len(faked), d_loss.item()/len(faked)), end='\r')
        p = np.random.permutation(len(data))
        data = data[p]
        print()
    torch.save(generator, "Generator.pth")
    torch.save(discriminator, "Discriminator.pth")
    genD = genD.data.cpu().numpy().squeeze()
    labs = labs.data.cpu().numpy()
    fig, ax = plt.subplots(8,4)
    ax = ax.ravel()
    for i in range(len(genD)):
        ax[i].plot(genD[i,:])
        ax[i].set_title(labs[i,-1])
    plt.show()
