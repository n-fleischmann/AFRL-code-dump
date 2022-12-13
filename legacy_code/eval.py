import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict

import pytorch_ood
from pytorch_ood.utils import is_known

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval(
    model: nn.Module, test_loader: DataLoader, detector: pytorch_ood.detector, limit=20
) -> Dict:
    """Evaluate the model using the given dataloader

    Args:
        model (nn.Module): Model to Evaluate
        test_loader (Dataloader): _description_
        detector (pytorch_ood.Detector, optional): pytorch OOD detector. Defaults to DETECTOR.
        limit (int, optional): Number of batches to evaluate on. Defaults to 20.

    Returns:
        Dict: dictionary of metrics
    """
    model.eval()
    model_detector = detector(model)

    metrics = pytorch_ood.utils.OODMetrics()

    with torch.no_grad():
        for i, (pkg) in enumerate(test_loader):
            if limit != -1 and i >= limit:
                break
            X, y = pkg[0].to(DEVICE), pkg[1].to(DEVICE)

            metrics.update(model_detector(X), y)

        return metrics.compute()


def ID_accuracy(model: nn.Module, mixed_loader: DataLoader, limit: int = 20) -> float:
    """Calculate the in-distribution accuracy from a loader of mixed ID/OOD data.

    Args:
        model (nn.Module): _description_
        mixed_loader (DataLoader): _description_
        limit (int, optional): _description_. Defaults to 20.

    Returns:
        float: ID accuracy in [0, 100]
    """
    running_correct = 0.0
    running_total = 0.0

    with torch.no_grad():
        for i, (pkg) in enumerate(mixed_loader):
            if limit != -1 and i >= limit:
                break
            X, y = pkg[0].to(DEVICE), pkg[1].to(DEVICE)

            outputs = model(X)
            _, preds = outputs.max(dim=1)

            running_correct += (preds.eq(y) * is_known(y)).sum().item()
            running_total += is_known(y).sum().item()

    return running_correct / running_total * 100
