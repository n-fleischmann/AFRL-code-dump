import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utilsdata
from torch.optim import SGD

import fgcv_aircraft_code.helpers as helpers
from data import rand_bbox, classes_converter
from legacy_code.eval import eval
from legacy_code.save_manager import SaveManager

import pytorch_ood
from pytorch_ood.utils import is_known
from pytorch_ood.detector import Softmax

import pynvml


import time
import os
import numpy as np

learning_rate_decay_schedule = [30, 60, 90]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class StandardTrainer:
    def __init__(self, model, objective, n_epochs, starting_epoch=1, rank=None):
        self.model = model
        self.objective = objective
        self.n_epochs = n_epochs
        self.starting_epoch = starting_epoch
        self.cur_epoch = None
        self.rank=rank

        self.print_template = "Epoch [ {:03} / {} ] Train ID Accuracy: {:3.3f}%  Epoch Loss: {:.6f}  Time:{:.3f}s  Avg Memory Usage: {}"

    def initialize_optimizer(self, lr, momentum, weight_decay, gamma):
        self.optimizer = SGD(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        self.learning_rate_table = helpers.create_learning_rate_table(
            lr, learning_rate_decay_schedule, gamma, self.n_epochs + 1
        )

    def train(self, train_loader_fn, test_loader_fn, save_manager, distributed=False):

        ID_loader, OOD_loader = train_loader_fn()

        testloader = test_loader_fn()

        pynvml.nvmlInit()

        for epoch in range(self.starting_epoch, self.n_epochs + 1):
            self.cur_epoch = epoch
            epoch_start_time = time.time()
            helpers.adjust_learning_rate(
                self.optimizer, epoch, self.learning_rate_table
            )

            train_acc, train_loss = self.train_once(ID_loader, OOD_loader, distributed=False)

            save_manager.save(self.model, epoch)

            metrics = eval(
                self.model, testloader, Softmax, limit=min(40, len(testloader))
            )

            memory_used = 0
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_used += pynvml.nvmlDeviceGetMemoryInfo(handle).used
            avg_memory_used = memory_used / device_count

            print_metrics = " | L1 AUCROC: {:3.3f} L1 Acc@95TPR: {:3.3f} L1 FPR@95TPR: {:3.3f}".format(
                metrics["AUROC"] * 100,
                metrics["ACC95TPR"] * 100,
                metrics["FPR95TPR"] * 100,
            )

            if self.rank is None or self.rank == 0: print(
                self.print_template.format(
                    epoch,
                    self.n_epochs,
                    train_acc,
                    train_loss,
                    time.time() - epoch_start_time,
                    avg_memory_used
                )
                + print_metrics
            )

        return self.model


    def train_once(self, ID_loader, OOD_loader, distributed=False):

        self.model.train()

        if distributed:
            ID_loader.sampler.set_epoch(self.cur_epoch)
            OOD_loader.sampler.set_epoch(self.cur_epoch)

        running_correct = 0.0
        running_total_known = 0.0
        running_total = 0.0
        running_loss_sum = 0.0

        for X, y in ID_loader:

            oe_X, oe_y = next(OOD_loader)

            if self.rank is not None:
                X, y = X.to(DEVICE + f":{self.rank}"), y.to(DEVICE + f":{self.rank}")
                oe_X, oe_y = oe_X.to(DEVICE + f":{self.rank}"), oe_y.to(DEVICE + f":{self.rank}")
            else:
                X, y = X.to(DEVICE), y.to(DEVICE)
                oe_X, oe_y = oe_X.to(DEVICE), oe_y.to(DEVICE)

            X, y = torch.cat((X, oe_X)), torch.cat((y, oe_y))

            outputs = self.model(X)
            loss = self.objective(outputs, y)
            _, preds = outputs.max(dim=1)

            running_correct += (preds.eq(y) * is_known(y)).sum().item()

            self.optimizer.zero_grad()
            self.model.zero_grad()

            running_total_known += is_known(y).shape[0]
            running_total += y.shape[0]

            running_loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()

        return (
            running_correct / running_total_known * 100,
            running_loss_sum / running_total * 1000,
        )


soft_xent = SoftCE()



class VanillaTrainer(StandardTrainer):

    def train_once(self, ID_loader, _, distributed=False):

        self.model.train()

        if distributed:
            ID_loader.sampler.set_epoch(self.cur_epoch)

        running_correct = 0.0
        running_total_known = 0.0
        running_total = 0.0
        running_loss_sum = 0.0

        for X, y in ID_loader:

            X, y = X.to(DEVICE), y.to(DEVICE)

            outputs = self.model(X)
            loss = self.objective(outputs, y)
            _, preds = outputs.max(dim=1)

            running_correct += (preds.eq(y) * is_known(y)).sum().item()

            self.optimizer.zero_grad()
            self.model.zero_grad()

            running_total_known += is_known(y).shape[0]
            running_total += y.shape[0]

            running_loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()

        return (
            running_correct / running_total_known * 100,
            running_loss_sum / running_total * 1000,
        )

        





class MixupTrainer(StandardTrainer):
    MIXUP_ALPHA = 2.0
    MIXUP_BETA = 5.0

    def __init__(self, model, objective, n_epochs, split, mix_op, starting_epoch=1, rank = None):
        self.model = model
        self.objective = objective
        self.n_epochs = n_epochs
        self.split = split
        self.mix_op = mix_op
        self.starting_epoch = starting_epoch
        self.cur_epoch = None
        self.rank = rank

    def train_once(self, ID_loader, OOD_loader, distributed=False):

        running_correct = 0.0
        running_total = 0.0
        running_loss_sum = 0.0

        self.model.train()

        if distributed:
            ID_loader.sampler.set_epoch(self.cur_epoch)
            OOD_loader.sampler.set_epoch(self.cur_epoch)

        N_ID_CLASSES = classes_converter[self.split]


        for X, y in ID_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            oe_X, _ = next(OOD_loader)
            oe_X = oe_X.to(DEVICE)

            one_hot_y = torch.zeros(X.shape[0], N_ID_CLASSES).to(DEVICE)
            one_hot_y.scatter_(1, y.view(-1, 1), 1)

            # ID Loss
            logits = self.model(X)
            _, preds = logits.softmax(dim=1).max(dim=1)
            id_loss = F.cross_entropy(logits, y)

            running_correct += (preds.eq(y) * is_known(y)).sum().item()
            running_total += y.shape[0]

            # MixOE Loss
            lam = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA)

            if self.mix_op == "cutmix":
                mixed_x = X.clone().detach()
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                # adjust lambda to exactly match pixel ratio
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
                )
                # we empirically find that pasting outlier patch into ID data performs better
                # than pasting ID patch into outlier data
                mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_X[:, :, bbx1:bbx2, bby1:bby2]
            elif self.mix_op == "mixup":
                mixed_x = lam * X + (1 - lam) * oe_X
            else:
                raise NotImplementedError

            oe_y = torch.ones(oe_X.size(0), N_ID_CLASSES).to(DEVICE) / N_ID_CLASSES
            mixed_y = lam * one_hot_y + (1 - lam) * oe_y

            mixed_loss = soft_xent(self.model(mixed_x), mixed_y)

            # OOD Loss
            #  oe_loss = F.cross_entropy(self.model(oe_x), oe_y)

            loss = id_loss + self.MIXUP_BETA * mixed_loss

            self.optimzer.zero_grad()
            self.model.zero_grad()

            running_loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()

        return (
            running_correct / running_total * 100,
            running_loss_sum / running_total * 1000,
        )
