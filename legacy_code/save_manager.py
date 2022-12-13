import torch
import os
import glob

from typing import Tuple


class SaveManager:
    """A Manager for saving/loading Models"""

    def __init__(
        self, root: os.PathLike, save_freq: int, n_epochs: int, experiment: int, rank: int = None
    ) -> None:
        """Create a SaveManager

        Args:
            root (os.PathLike): The subfolder of ./models to save/load to/from
            save_freq (int): Frequency of model saves
            n_epochs (int): Number of epochs in training
            experiment (int): Number of this experiment
        """
        self.root = root
        self.save_freq = save_freq
        self.n_epochs = n_epochs
        self.rank = rank

        self.experiment = experiment
        self.save_path = os.path.join(self.root, str(self.experiment))

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def save(self, model: torch.nn.Module, epoch: int) -> None:
        """Save the current model

        Args:
            model (torch.nn.Module): Save me!
            epoch (int): current epoch
        """

        if self.rank is None or self.rank == 0:
            if (
                epoch % self.save_freq == 0 and self.save_freq != -1
            ) or epoch == self.n_epochs:
                path = os.path.join(self.save_path, f"epoch_{epoch}.pt")
            else:
                path = os.path.join(self.save_path, "tmp.pt")

            torch.save(
                model.state_dict, path
            )

    def load(self, model: torch.nn.Module, epoch: int) -> Tuple[torch.nn.Module, int]:
        """Load the parameters from the given epoch into the model

        Args:
            model (torch.nn.Module): Load me
            epoch (int): the epoch to load from

        Returns:
            Tuple[torch.nn.Module, int]: the model back, along with the starting epoch for training
        """
        model.load_state_dict(
            torch.load(
                os.path.join(self.root, str(self.experiment), f"epoch_{epoch}.pt")
            )()
        )

        print(f"Epoch {epoch} successfully loaded.")

        return model, epoch + 1

    @staticmethod
    def _get_epoch(path: str) -> int:
        """Disect the epoch from the full model path

        Args:
            path (str): model path

        Returns:
            int: epoch
        """
        return int(path.split("/")[-1].replace(".pt", "").replace("epoch_", ""))

    def find_and_load(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Wrapper for load that finds the last epoch in the current folder

        Args:
            model (torch.nn.Module): the model to load from

        Returns:
            Tuple[torch.nn.Module, int]: see self.load
        """

        params = glob.glob(os.path.join(self.root, str(self.experiment), "epoch_*.pt"))
        highest_epoch = self._get_epoch(sorted(params, key=self._get_epoch)[-1])

        return self.load(model, highest_epoch)
