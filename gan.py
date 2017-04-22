import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os


# hyperparameters

lr = 1e-3
batch_size = 16
H = 100
D_img = 28*28
D_labels = 10
D_ent = 100
print_every = 1000
global_step = 0
iters = 50000

d_steps = 1
g_steps = 1


# dataloader

modes = ['train', 'val']
trans = transforms.Compose([transforms.ToTensor(),]) # transforms.Normalize((0.1307,), (0.3081,))
dsets = {k: datasets.MNIST('./data', train=k=='train', download=True, transform=trans) for k in modes}
loaders = {k: torch.utils.data.DataLoader(dsets[k], batch_size=batch_size, shuffle=True) for k in modes}

dset_sizes = {k: len(dsets[k]) for k in modes}
dset_dims = dsets['train'].train
dset_classes = [str(x) for x in range(D_labels)]


# visualize data

def imsave(samples, c, title=None):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
    plt.close(fig)

inputs, classes = next(iter(loaders['train']))
samples = inputs.numpy()[:16]
imsave(samples, 0)


# G and D networks

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_3 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        return F.sigmoid(self.linear_3(x))

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_3 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        return F.sigmoid(self.linear_3(x))

D = Discriminator(input_size=D_img, hidden_size=H, output_size=1)
G = Generator(input_size=D_ent, hidden_size=H, output_size=D_img)
criterion = nn.BCELoss()  # binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
D_optimizer = optim.Adam(D.parameters(), lr=lr)
G_optimizer = optim.Adam(G.parameters(), lr=lr)


# train networks

ones_label = Variable(torch.ones(batch_size,1))
zeros_label = Variable(torch.zeros(batch_size,1))

D_loss_avg = G_loss_avg = None ; interp = 0.99
for global_step in range(global_step+1,global_step+1+iters):
    # Sample data
    z = Variable(torch.randn(batch_size, D_ent))
    X, _ = next(iter(loaders['train']))
    X = Variable(X).resize(batch_size,D_img)

    # Dicriminator forward-loss-backward-update
    G_sample = G(z)
    D_real = D(X) # .resize(batch_size,1,28,28)
    D_fake = D(G_sample)

    D_loss_real = criterion(D_real, ones_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss = .5*(D_loss_real + D_loss_fake)

    D_loss.backward()
    D_optimizer.step()
    D.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(batch_size, D_ent))
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_optimizer.step()
    G.zero_grad()
    
    D_loss_avg = D_loss if D_loss_avg is None else interp*D_loss_avg + (1-interp)*D_loss
    G_loss_avg = G_loss if G_loss_avg is None else interp*G_loss_avg + (1-interp)*G_loss
    
    if global_step % print_every == 0:
        print("step %s: d_loss: %s and g_loss: %s"% (global_step,
                                                     D_loss_avg.data.numpy()[0],
                                                     G_loss_avg.data.numpy()[0]))

        samples = G(z).data.numpy()[:16]
        c = global_step // print_every
        imsave(samples, c)