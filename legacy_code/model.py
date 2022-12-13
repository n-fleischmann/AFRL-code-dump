import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights

from typing import List

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_model(pretrained, n_id_classes, parallel: bool = True, gpu_priority: List = None) -> nn.Module:
    """Create a Resnet 50 Model

    Args:
        parallel (bool, optional): If true, wrap model in an nn.DataParallel. Defaults to True.
        gpu_priority (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, n_id_classes)
    if parallel:
        model = nn.DataParallel(model, device_ids=gpu_priority)
    return model