import torch
from torch import nn
from models import Generator, Discriminator, GeneratorLoss

k = 1
lr = 0.01

classical_to_humming_gen = Generator()
humming_to_classical_gen = Generator()
classical_disc = Discriminator()
humming_disc = Discriminator()

classical_to_humming_loss = GeneratorLoss(humming_disc)
humming_to_classical_loss = GeneratorLoss(classical_disc)
classical_disc_loss = nn.BCEwithLogitsLoss()
humming_disc_loss = nn.BCEwithLogitsLoss()

classical_to_humming_optim = 
humming_to_classical_optim = 
classical_disc_optim = 
humming_disc_optim = 

epochs = 10
for epoch in epochs:
    for disc_updates in range(k):
        classical_logits = Discriminator()
        humming_logits = Discriminator()
        