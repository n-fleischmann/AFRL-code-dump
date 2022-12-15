from datetime import datetime
import sched
from tabnanny import verbose
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from data import combined_loader, ternary_loader, training_data

import pytorch_lightning as pl
import pytorch_ood
from pytorch_ood.utils import is_known, is_unknown, OODMetrics

import torchmetrics

from typing import Dict, List
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def create_model(pretrained, n_id_classes, parallel: bool = True, gpu_priority: List = None) -> nn.Module:
    """Create a Resnet 50 Model

    Args:
        parallel (bool, optional): If true, wrap model in an nn.DataParallel. Defaults to True.
        gpu_priority (_type_, optional): _description_. Defaults to None.

    Returns:
        nn.Module
    """
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, n_id_classes)
    if parallel:
        model = nn.DataParallel(model, device_ids=gpu_priority)
    return model






def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    TODO: use bisect

    :param pred: outlier scores
    :param target: target label
    :param k: cutoff value
    :return:
    """
    fpr, tpr, thresholds = torchmetrics.functional.roc(pred, target)
    for fp, tp, t in zip(fpr, tpr, thresholds):
        if tp >= k:
            return fp

    return torch.tensor(1.0)



class GenericLightningModel(pl.LightningModule):

    def __init__(self, config: Dict[str, object]):
        super().__init__()

        self.config = config

        self.n_id_classes = config["n_id_classes"]
        self.objective = config["objective"]
        self.max_epochs = config["max_epochs"]
        self.model = create_model(pretrained=True, n_id_classes=self.n_id_classes, parallel=False)

        self.last_epoch_end_time = time.time()

        self.epoch_print_template = "[ {} ] Epoch [ {:03} / {} ] Training ID Accuracy: {:3.3f}% Epoch Loss: {:.6f} Time {:.3f}s"
        self.validation_print_template = "\t L1 AUROC {:.4f} L1 FPR95TPR {:.4f}"


    def forward(self, X):
        return self.model(X)


    def get_training_loader(self):
        return combined_loader(
            self.config['dataset'], 
            self.config['split'], 
            self.config['batch_size'], 
            self.config['ood_ratio'], 
            num_workers=(18, 18)
        )


    def training_step(self, batch, batch_idx):

        loss = None
        ID_correct = None
        N_ID = None

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID}


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, verbose=True)

        return [optimizer], [scheduler]


    def training_epoch_end(self, outputs):

        gathered = self.all_gather(outputs)
        if self.global_rank == 0:

            loss = sum(output['loss'].mean().detach() for output in gathered) / len(outputs)
            
            correct = sum(output['ID_correct'].sum() for output in gathered)
            total_id = sum(output['N_ID'].sum() for output in gathered)

            ID_accuracy = correct / total_id * 100

            self.print(
                self.epoch_print_template.format(
                    datetime.now(),
                    self.current_epoch + 1, 
                    self.max_epochs, 
                    float(ID_accuracy), 
                    float(loss), 
                    time.time() - self.last_epoch_end_time
                )
            )

            self.last_epoch_end_time = time.time()


    def validation_step(self, batch, _):
        X, y= batch
        return {"scores": -self.model(X).softmax(dim=1).max(dim=1).values, "targets": y}


    def validation_epoch_end(self, val_epoch_outputs):

        gathered = self.all_gather(val_epoch_outputs)
        if self.global_rank == 0:

            scores = torch.stack([output['scores'] for output in gathered])
            targets = torch.stack([output['targets'] for output in gathered])

            label = is_unknown(targets).detach().to(self.device).long()

            AUROC = torchmetrics.AUROC(num_classes=2)(scores, label)
            FPR = fpr_at_tpr(scores, targets)

            self.print(self.validation_print_template.format(float(AUROC) * 100, float(FPR) * 100))



class OEModel(GenericLightningModel):


    def training_step(self, batch, _):
        X, y = batch["ID"]
        oe_X, oe_y = batch['OOD']

        X, y = torch.cat((X, oe_X)), torch.cat((y, oe_y))

        outputs = self.model(X)
        loss = self.objective(outputs, y)

        _, preds = outputs.softmax(dim=1).max(dim=1)

        ID_correct = (preds.eq(y) * is_known(y)).sum().detach()
        N_ID = is_known(y).sum().detach()

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID}



class VanillaModel(GenericLightningModel):

    def get_training_loader(self):
        training_loader, _ = training_data(self.config['dataset'], self.config['split'], self.config['batch_size'], 'big', 1, num_workers=(36, 0))
        return training_loader


    def training_step(self, batch, _):
        X, y = batch
        outputs = self.model(X)
        loss = self.objective(outputs, y)

        _, preds = outputs.softmax(dim=1).max(dim=1)
        ID_correct = (preds.eq(y) * is_known(y)).sum().detach()
        N_ID = is_known(y).sum().detach()

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID}


class SoftCE(nn.Module):
    def __init__(self, reduction="mean"):
        super(SoftCE, self).__init__()
        self.reduction = reduction


    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(
                "Reduction type '{:s}' is not supported!".format(self.reduction)
            )




def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




class MixupModel(GenericLightningModel):
    MIXUP_ALPHA = 2.0

    def __init__(self, config):

        super().__init__(config)
        self.mixup_mode = config["mixup_mode"]
        self.MIXUP_BETA = config["mixup_beta"]
        self.soft_xent = SoftCE()


    def get_training_loader(self):
        return combined_loader(self.config['dataset'], self.config['split'], self.config['batch_size'], 1, num_workers=(1, 1))


    def training_step(self, batch, batch_idx):

        X, y = batch["ID"]
        oe_X, oe_y = batch['OOD']
        oe_X, oe_y = oe_X[ : X.shape[0]], oe_y[: X.shape[0]]

        one_hot_y = torch.zeros(X.shape[0], self.n_id_classes, device=self.device)
        one_hot_y = torch.scatter(one_hot_y, 1, y.view(-1, 1), 1)

        # ID Loss
        logits = self.model(X)
        _, preds = logits.softmax(dim=1).max(dim=1)
        id_loss = F.cross_entropy(logits, y)

        ID_correct = (preds.eq(y) * is_known(y)).sum().detach()
        N_ID = is_known(y).sum().detach()

        # MixOE Loss - Taken from their REPO
        lam = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA)

        if self.mixup_mode == "cutmix":
            mixed_x = X.clone().detach()
            bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)

            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
            )

            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_X[:, :, bbx1:bbx2, bby1:bby2]
        elif self.mixup_mode == "mixup":
            mixed_x = lam * X + (1 - lam) * oe_X
        else:
            raise NotImplementedError

        oe_y = torch.ones(oe_X.size(0), self.n_id_classes, device=self.device) / self.n_id_classes
        mixed_y = lam * one_hot_y + (1 - lam) * oe_y
        mixed_loss = self.soft_xent(self.model(mixed_x), mixed_y)

        loss = id_loss + self.MIXUP_BETA * mixed_loss

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID} 
        



class TernaryModel(MixupModel):
    MIXUP_ALPHA = 2.0
    MIXUP_BETA = 5.0

    def __init__(self, config):
        super().__init__(config)
        self.MIXUP_GAMMA = config['ternary_gamma']

    
    def get_training_loader(self):
        return ternary_loader(self.config['dataset'], self.config['split'], self.config['batch_size'], 1, num_workers=(1, 1, 1))


    def training_step(self, batch, _):

        ret_dict = super().training_step(batch, _)

        X, y = batch["ID"] # True ID Training sample
        supp_x, supp_y = batch['SUPP_ID'] # Supplemental ID from infinite generator

        one_hot_y = torch.zeros(X.shape[0], self.n_id_classes, device=self.device)
        one_hot_y = torch.scatter(one_hot_y, 1, y.view(-1, 1), 1)

        lam_id_id = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA)

        if self.mixup_mode == 'cutmix':
            mix_id_id_x = X.clone().detach()
            bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam_id_id)
            lam_id_id = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
            )

            mix_id_id_x[:, :, bbx1:bbx2, bby1:bby2] = supp_x[:, :, bbx1:bbx2, bby1:bby2]
        elif self.mixup_mode == 'mixup':
            mix_id_id_x = lam_id_id * X + (1 - lam_id_id) * supp_x
        else:
            raise NotImplementedError

        tmp = torch.zeros(X.shape[0], self.n_id_classes, device=self.device)
        id_id_y = one_hot_y * lam_id_id + torch.scatter(tmp, 1, supp_y.view(-1, 1), 1.0 - lam_id_id)
        mixed_id_id_loss = self.soft_xent(self.model(mix_id_id_x), id_id_y)

        ret_dict["loss"] += self.MIXUP_GAMMA * mixed_id_id_loss

        return ret_dict




class QuadModel(TernaryModel):

    def __init__(self, config):
        super().__init__(config)
        self.MIXUP_DELTA = config['quad_delta']

    def training_step(self, batch, _):
        ret_dict = super().training_step(batch, _)
        oe_X, oe_y = batch['OOD'] # OOD Sample

        ret_dict['loss'] += self.MIXUP_DELTA * pytorch_ood.loss.OutlierExposureLoss()(self.model(oe_X), oe_y)

        return ret_dict
