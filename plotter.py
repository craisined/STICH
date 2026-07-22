from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Plotter:
  def _plotLoss(self, id, classical_disc_loss, humming_disc_loss, humming_to_classic_gen_loss, classical_to_humming_gen_loss):
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{id}.png"
    
    plt.ylabel("Loss")
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.plot(classical_disc_loss, label="Classical Discriminator")
    plt.plot(humming_disc_loss, label="Humming Discriminator")
    plt.plot(humming_to_classic_gen_loss, label="Humming to Classical Generator")
    plt.plot(classical_to_humming_gen_loss, label="Classical to Humming Generator")
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
  def plotEpochLoss(self, epoch, classical_disc_loss, humming_disc_loss, humming_to_classic_gen_loss, classical_to_humming_gen_loss):
    plt.title(f"Loss Over Iteration: Epoch {epoch}")
    plt.xlabel("Iteration")
    
    self._plotLoss(f"epoch{epoch}", classical_disc_loss, humming_disc_loss, humming_to_classic_gen_loss, classical_to_humming_gen_loss)
    
  def plotFullLoss(self, classical_disc_loss, humming_disc_loss, humming_to_classic_gen_loss, classical_to_humming_gen_loss):
    plt.title(f"Loss Over Epoch")
    plt.xlabel("Epoch")
    
    self._plotLoss(f"full", classical_disc_loss, humming_disc_loss, humming_to_classic_gen_loss, classical_to_humming_gen_loss)