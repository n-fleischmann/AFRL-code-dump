from datetime import datetime
import sched
from tabnanny import verbose
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from data import rand_bbox

import pytorch_lightning as pl
import pytorch_ood
from pytorch_ood.utils import is_known, is_unknown, OODMetrics

import torchmetrics

from typing import List
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

    def __init__(self, n_id_classes, objective, max_epochs, detector):
        super().__init__()

        self.n_id_classes = n_id_classes
        self.objective = objective
        self.max_epochs = max_epochs
        self.model = create_model(pretrained=True, n_id_classes=n_id_classes, parallel=False)

        self.last_epoch_end_time = time.time()

        self.epoch_print_template = "[ {} ] Epoch [ {:03} / {} ] Training ID Accuracy: {:3.3f}% Epoch Loss: {:.6f} Time {:.3f}s"
        self.validation_print_template = "\t L1 AUROC {:.4f} L1 FPR95TPR {:.4f}"


    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):

        loss = None
        ID_correct = None
        N_ID = None

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, verbose=True)

        # return [optimizer], [{'scheduler': scheduler, "interval": "epoch"}]
        return [optimizer], [scheduler]


    def training_epoch_end(self, outputs):

        gathered = self.all_gather(outputs)
        if self.global_rank == 0:

            # loss = torch.stack([torch.stack([x['loss'] for x in output] for output in gathered)]).mean()
            loss = sum(output['loss'].mean().detach() for output in gathered) / len(outputs)
            
            correct = sum(output['ID_correct'].sum() for output in gathered)
            total_id = sum(output['N_ID'].sum() for output in gathered)

            ID_accuracy = correct / total_id * 100

            # ID_accuracy = sum(output['ID_correct'].sum().detach() for output in gathered) / sum(output["N_ID"].sum().detach() for output in gathered) * 100

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

        # if self.global_rank == 0:
        #     X, y = batch
        #     self.metric_holder.update(self.model_detector(X), y)


    def validation_epoch_end(self, val_epoch_outputs):

        gathered = self.all_gather(val_epoch_outputs)
        if self.global_rank == 0:

            scores = torch.stack([output['scores'] for output in gathered])
            targets = torch.stack([output['targets'] for output in gathered])

            label = is_unknown(targets).detach().to(self.device).long()

            AUROC = torchmetrics.AUROC(num_classes=2)(scores, label)
            FPR = fpr_at_tpr(scores, targets)

            self.print(self.validation_print_template.format(float(AUROC) * 100, float(FPR) * 100))



            # metrics = self.metric_holder.compute()
            # self.print("\t L1 AUROC: {:.5f} L1 FPR95TPR: {:.5f}".format(metrics['AUROC'] * 100, metrics['FPR95TPR'] * 100))

            # self.metric_holder.reset()



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



class MixupModel(GenericLightningModel):
    MIXUP_ALPHA = 2.0
    MIXUP_BETA = 5.0

    def __init__(self, n_id_classes, objective, max_epochs, mixup_mode, detector):

        super().__init__(n_id_classes, objective, max_epochs, detector)
        self.mixup_mode = mixup_mode
        self.soft_xent = SoftCE()

    def training_step(self, batch, batch_idx):

        X, y = batch["ID"]
        oe_X, oe_y = batch['OOD']
        oe_X, oe_y = oe_X[ : X.shape[0]], oe_y[: X.shape[0]]

        one_hot_y = torch.zeros(X.shape[0], self.n_id_classes, device=self.device)
        one_hot_y = torch.scatter(one_hot_y, 1, y.view(-1, 1), 1)
        # one_hot_y.scatter_(1, y.view(-1, 1), 1)

        # ID Loss
        logits = self.model(X)
        _, preds = logits.softmax(dim=1).max(dim=1)
        id_loss = F.cross_entropy(logits, y)

        ID_correct = (preds.eq(y) * is_known(y)).sum().detach()
        N_ID = is_known(y).sum().detach()

        # MixOE Loss
        lam = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA)

        if self.mixup_mode == "cutmix":
            mixed_x = X.clone().detach()
            bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
            )
            # we empirically find that pasting outlier patch into ID data performs better
            # than pasting ID patch into outlier data
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

    def set_ternary_gamma(self, gamma):
        assert isinstance(gamma, float)
        assert gamma > 0, f"Ternary Gamma must be positive float, got {gamma}"
        self.MIXUP_GAMMA = gamma


    def training_step(self, batch, _):
        X, y = batch["ID"] # True ID Training sample
        oe_X, oe_y = batch['OOD'] # OOD Sample
        supp_x, supp_y = batch['SUPP_ID'] # Supplemental ID from infinite generator

        oe_X, oe_y = oe_X[ : X.shape[0]], oe_y[: X.shape[0]]
        supp_x, supp_y = supp_x[ : X.shape[0]], supp_y[: X.shape[0]]

        one_hot_y = torch.zeros(X.shape[0], self.n_id_classes, device=self.device)
        one_hot_y = torch.scatter(one_hot_y, 1, y.view(-1, 1), 1)
        # one_hot_y.scatter_(1, y.view(-1, 1), 1)

        # ID Loss
        logits = self.model(X)
        _, preds = logits.softmax(dim=1).max(dim=1)

        id_loss = F.cross_entropy(logits, y)

        ID_correct = (preds.eq(y) * is_known(y)).sum().detach()
        N_ID = is_known(y).sum().detach()

        # ID/OOD Mix Loss

        lam_id_ood = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA)

        if self.mixup_mode == 'cutmix':
            mixed_x = X.clone().detach()
            bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam_id_ood)

            lam_id_ood = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
            )

            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_X[:, :, bbx1:bbx2, bby1:bby2]
        elif self.mixup_mode == "mixup":
            mixed_x = lam_id_ood * X + (1 - lam_id_ood) * oe_X
        else:
            raise NotImplementedError


        oe_y = torch.ones(oe_X.size(0), self.n_id_classes, device=self.device) / self.n_id_classes
        mixed_y = lam_id_ood * one_hot_y + (1 - lam_id_ood) * oe_y
        mixed_oe_id_loss = self.soft_xent(self.model(mixed_x), mixed_y)


        # ID/ID Mix Loss 

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

        loss = id_loss + self.MIXUP_BETA * mixed_oe_id_loss + self.MIXUP_GAMMA * mixed_id_id_loss

        return {"loss": loss, "ID_correct": ID_correct, "N_ID": N_ID} 




class QuadModel(TernaryModel):

    def set_delta(self, delta):
        assert isinstance(delta, float)
        assert delta > 0, f"OOD Weighting must be positive float, got {delta}"
        self.MIXUP_DELTA = delta

    
    def training_step(self, batch, _):
        ret_dict = super().training_step(batch, _)
        oe_X, oe_y = batch['OOD'] # OOD Sample

        ret_dict['loss'] += self.MIXUP_DELTA * pytorch_ood.loss.OutlierExposureLoss()(self.model(oe_X), oe_y)

        return ret_dict
