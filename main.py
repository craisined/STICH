import torch
from torch import nn
from models import Generator, Discriminator, GeneratorLoss
import logging

logger = logging.getLogger(__name__)

k = 1
lr = 0.01

classical_to_humming_gen = Generator()
humming_to_classical_gen = Generator()
classical_disc = Discriminator()
humming_disc = Discriminator()

classical_to_humming_loss = GeneratorLoss(humming_disc, humming_to_classical_gen)
humming_to_classical_loss = GeneratorLoss(classical_disc, classical_to_humming_gen)
classical_disc_loss = nn.BCEwithLogitsLoss()
humming_disc_loss = nn.BCEwithLogitsLoss()

classical_to_humming_optim = torch.optim.SGD(classical_to_humming_gen.parameters(), lr=lr)
humming_to_classical_optim = torch.optim.SGD(humming_to_classical_gen.parameters(), lr=lr)
classical_disc_optim = torch.optim.SGD(classical_disc.parameters(), lr=lr)
humming_disc_optim = torch.optim.SGD(humming_disc.parameters(), lr=lr)

epochs = 10
for epoch in epochs:

    logger.info(f"Epoch {epoch}: Train discriminator")
    for disc_updates in range(k):

        # TODO: inputs and randomize
        classical_output = humming_to_classical_gen()
        humming_output = classical_to_humming_gen()

        classical_logits = classical_disc(classical_output)
        humming_logits = humming_disc(humming_output)

        classical_loss_val = classical_disc_loss(classical_logits)
        humming_loss_val = humming_disc_loss(humming_logits)

        classical_loss_val.backward()
        humming_loss_val.backward()
        classical_disc_optim.step()
        humming_disc_optim.step()

        logger.info(f"Loss for classical discriminator: {classical_loss_val}")
        logger.info(f"Loss for humming discriminator: {humming_loss_val}")

    logger.info(f"Epoch {epoch}: Train generator")

    # TODO: inputs
    classical_output = humming_to_classical_gen()
    humming_output = classical_to_humming_gen()

    classical_loss_val = humming_to_classical_loss(classical_output)
    humming_loss_val = classical_to_humming_loss(humming_output)

    classical_loss_val.backward()
    humming_loss_val.backward()
    classical_to_humming_optim.step()
    humming_to_classical_optim.step()

    logger.info(f"Loss for classical discriminator: {classical_loss_val}")
    logger.info(f"Loss for humming discriminator: {humming_loss_val}")