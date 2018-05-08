import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np

class torchAutoEnc(nn.Module):
    def __init__(self, shape=(1, 300)):
        super(torchAutoEnc, self).__init__()

        # Creating model layers. Padding crap had to be figured out manually
        self.e1 = nn.Conv1d(1,10, kernel_size=9, padding=4)
        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool1d(2, return_indices=True)
        self.e2 = nn.Conv1d(10, 10, kernel_size=19, padding=10)
        self.r2 = nn.ReLU()
        self.m2 = nn.MaxPool1d(2, return_indices=True)
        self.h_l, self.u_l = self.get_dimension(shape)
        self.e3 = nn.Linear(self.h_l, 100)
        self.r3 = nn.ReLU()
        self.e4 = nn.Linear(100, 10)
        self.r4 = nn.ReLU()
        self.d4 = nn.Linear(10, 100)
        self.d4.weight = torch.nn.Parameter(self.e4.weight.t())
        self.dr4 = nn.ReLU()
        self.d3 = nn.Linear(100, self.h_l)
        self.d3.weight = torch.nn.Parameter(self.e3.weight.t())
        self.dr3 = nn.ReLU()
        self.u2 = nn.MaxUnpool1d(2)
        self.d2 = nn.Conv1d(10,10, kernel_size=10, padding=5)
        self.d2.weight = torch.nn.Parameter(self.e2.weight.t())
        self.dr2 = nn.ReLU()
        self.u1 = nn.MaxUnpool1d(2)
        self.d1 = nn.Conv1d(10, 1, kernel_size=20, padding=10)
        self.d1.weight = torch.nn.Parameter(self.e1.weight.t())
        self.dr1 = nn.ReLU()

        # Classification layer
        self.eClassify = nn.Linear(10, 2)
        self.erClassify = nn.Softmax()

    def get_dimension(self, shape):
        # Dynamically get shape of flattened layer
        bs = 1
        inn= Variable(torch.rand(bs, *shape))
        out, _ = self.m1(self.e1(inn).relu())
        out, _ = self.m2(self.e2(out).relu())
        n_size = out.data.view(bs, -1).size(1)
        return n_size, out.shape


    def forward(self, inn, mode="autoenc"):
        outE, ind = self.forwardEnc(inn)
        # partition for either forward function with classify or decoding
        if mode == "classify":
            out = self.forwardClassify(outE)
        else:
            out = self.forwardDec(outE, ind)
        return out

    def forwardClassify(self, cinn):
        out = self.erClassify(self.eClassify(cinn))
        return out

    def forwardEnc(self,inn):
        ind = []
        outconvs1, ind1 = self.m1(self.r1(self.e1(inn)))
        outconvs2, ind2 = self.m2(self.r2(self.e2(outconvs1)))
        # indices for unpooling later
        ind.append(ind1)
        ind.append(ind2)
        # Die flattening
        outflatten = outconvs2.view(outconvs2.size(0),-1)
        outlins = self.r4(self.e4(self.r3(self.e3(outflatten))))
        return outlins, ind

    def forwardDec(self, dinn, indices):
        # Use functional API for tying weights with encoding layer (form of
        # regularization: reduces parameter space by factor of 2, kills curse of
        # dimensionality. Main reason why I even decided to use pytorch over
        # keras. It's more clear and open to implement)

        outlins = F.linear(F.linear(dinn, self.e4.weight.t()).relu(), self.e3.weight.t())
        outreshape = outlins.reshape(outlins.size(0), self.u_l[1], self.u_l[2])
        outconvs1 = F.conv1d(input=self.u2(outreshape, indices[1]), weight=self.e2.weight.t(), padding=8).relu()
        outconvs2 = F.conv1d(input=self.u1(outconvs1, indices[0]), weight=self.e1.weight.t(), padding=4)
        return outconvs2

model = torchAutoEnc()

def trainModel(nEpochs, input_data, labels, validation, batch_size):
    print()
    print()
    print("epoch\tEin_sample\tEout_sample")
    b = 5
    # Convert our numpy types to torch types
    data = torch.from_numpy(input_data[0]).float()
    targetData = torch.from_numpy(input_data[1]).float()
    valData = torch.from_numpy(validation[0]).float()
    targetvalData= torch.from_numpy(validation[1]).float()
    labelTrain = torch.from_numpy(labels[0]).long()
    labelVal = torch.from_numpy(labels[1]).long()

    # Criterion for autoencoder and classifier respectively
    criterion = torch.nn.MSELoss(size_average=False)
    criterion1 = torch.nn.CrossEntropyLoss(size_average=False)

    # Optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    l = [1000]
    # First train encoder
    for i in range(nEpochs):
        j = 0
        while j*b< len(data) - b:

            # Forward batch
            y_pred = model(data[j*b:j*b+b], "autoenc")

            # Compute Cost
            loss = criterion(y_pred, data[j*b:j*b+b])

            # set grad zero before backprop
            optimizer.zero_grad()

            #Backprop
            loss.backward()
            optimizer.step()

            j+=1
            print(i, loss.item(), end='\r')
        j = 0
        # Adaptive minibatch
        if i%100 == 0:
            b *=5
            if b > batch_size:
                b = batch_size
        validate = (model.forward(valData, "autoenc") - valData).pow(2).sum().item()/len(valData)
        print(i, loss.item()/b, validate)

        # Reshuffle data
        p = np.random.permutation(len(input_data[0]))
        input_data = [input_data[0][p], input_data[1][p]]
        data = torch.from_numpy(input_data[0]).float()
        targetData = torch.from_numpy(input_data[1]).float()
        optimizer.zero_grad()

    # Train Classifier

    # Reinitialize optimizer. Adam is adaptive so want to clear history
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Fix weights of encoder
    model.e1.weight.require_grad=False
    model.e2.weight.require_grad=False
    model.e3.weight.require_grad=False
    model.e4.weight.require_grad=False
    # Reinitialize batch length
    b = 5

    for i in range(nEpochs):
        j = 0
        while j*b< len(data) - b:

            # Forward batch
            y_pred = model(data[j*b:j*b+b], "classify")

            # Compute Cost
            loss1 = criterion1(y_pred, labelTrain[j*b:j*b+b])

            # set grad zero before backprop
            optimizer.zero_grad()

            #Backprop
            loss1.backward()
            optimizer.step()

            j+=1
            print(i, loss.item(), end='\r')
        j = 0
        # Adaptive minibatch
        if i%100 == 0:
            b *=5
            if b > batch_size:
                b = batch_size
        # if you convert labelVal into one-hot matrix, this should be fine
        #validate = (model.forward(valData, "classify") - labelVal).pow(2).sum().item()/len(labelVal)
        print(i, loss.item()/b, validate)

        # Reshuffle data
        p = np.random.permutation(len(input_data[0]))
        labelTrain = torch.from_numpy(labels[0][p]).long()
        optimizer.zero_grad()
