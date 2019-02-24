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
        return out, labels

class rDiscriminator(nn.Module):
    def __init__(self, nF=1):
        super(rDiscriminator, self).__init__()

        self.discT = nn.GRU(nF+1,2,1, batch_first=True)

    def forward(self, data):
        hidden = None
        out, hidden = self.discT(data, hidden)

        #return last time step.
        return out[:,-1,:]

if __name__ == "__main__":
    # Hyperparms
    batch = 32
    epochs = 1000
    size = 600
    lr = 1e-3

    discriminator = rDiscriminator().cuda()
    generator = rGenerator().cuda()

    adversarialLoss = nn.CrossEntropyLoss(reduction='sum').cuda()
    discriminatorLoss = nn.CrossEntropyLoss(reduction='sum').cuda()

    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

    ### Put data loading code here ###

    # Shuffle Data #
    p = np.random.permutation(len(data))
    data = data[p]
    for i in range(epochs):
        for j in range(0, data.size(0), batch):
            # Fake or not tags for loss Funcs
            validLabs = torch.LongTensor([1 for k in range(len(data[j:j+batch]))]).cuda()
            fakeLabs = torch.LongTensor([0 for k in range(len(data[j:j+batch]))]).cuda()

            # Generator Stage #
            optimizerG.zero_grad()

            genD, labs = generator(size, len(data[j:j+batch]))
            faked = torch.cat((genD, labs.unsqueeze(2)), dim=2)
            fakeres = discriminator(faked)
            # Loss for how well we fool the discriminator
            lossG = adversarialLoss(fakeres, validLabs)

            lossG.backward()
            optimizerG.step()

            # Discriminator Stage #
            optimizerD.zero_grad()

            validres = discriminator(data[j:j+batch])
            fakeres = discriminator(faked.detach())

            # Loss for how well the discrminator can detect fakes
            lossDG = discriminatorLoss(fakeres, fakeLabs)
            lossDY = discriminatorLoss(validres, validLabs)

            d_loss = (lossDG + lossDY)/2
            d_loss.backward()
            optimizerD.step()
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
