import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as utilsdata
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder

import pytorch_ood

import time
from argparse import ArgumentParser
import os
from datetime import datetime
import glob
import sys

from fgcv_aircraft_code import Dataset_fromTxtFile as txtfiledataset
import fgcv_aircraft_code.helpers as helpers
from data import rand_bbox, training_data, testing_data

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


################################################################
#                    Command Line Arguments                    #
################################################################

DETECTOR_CONVERTER = {
    "odin": pytorch_ood.detector.ODIN,
    "softmax": pytorch_ood.detector.Softmax,
    "energy": pytorch_ood.detector.NegativeEnergy,
}

OBJECTIVE_CONVERTER = {
    "ring": pytorch_ood.loss.ObjectosphereLoss(alpha=0.0001, xi=50),
    "eosl": pytorch_ood.loss.EntropicOpenSetLoss(),
    "oe": pytorch_ood.loss.OutlierExposureLoss(),
    "energy": pytorch_ood.loss.EnergyRegularizedLoss(
        alpha=0.1, margin_in=27, margin_out=9
    ),
    "cross_entropy": pytorch_ood.loss.CrossEntropyLoss(),
    "mixup": None,
}


parser = ArgumentParser()

parser.add_argument(
    "split", default=1, type=int, help="ID/OOD Split to use. One of [1, 2, 3, 4]."
)
parser.add_argument("n_experiments", type=int, help="Number of models to train")


parser.add_argument("objective", choices=OBJECTIVE_CONVERTER.keys())

parser.add_argument(
    "--detector", "-d", choices=DETECTOR_CONVERTER.keys(), default="softmax"
)

parser.add_argument(
    "--mode",
    "-m",
    default="traineval",
    choices=["train", "traineval", "eval"],
    help="Training/Evaluating mode. Beware memory usage of training and \
        testing set when using traineval on large datasets. [Default: traineval]",
)
parser.add_argument(
    "--imsize",
    default="big",
    choices=["big", "medium", "tiny"],
    help="Image size: big 448x448, medium 224x224, tiny 32x32. [Default: medium]",
)
parser.add_argument(
    "--not_pretrained",
    action="store_false",
    help="Turn OFF ImageNet pretrained weights.",
)
parser.add_argument("--mix_op", choices=["mixup", "cutmix"], default="cutmix")
parser.add_argument("--batch_size", "-b", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument(
    "--save_folder", help="Subfolder in './models/ to save/load models to/from"
)
parser.add_argument(
    "--checkpoint",
    type=int,
    help="Epoch checkpoint to load for training/evaluating. Will throw an error if save_folder is not specified.",
)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.00001)
parser.add_argument("--gamma", type=float, default=0.1)

parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Random seed for model initialization and random shuffling.",
)

args = parser.parse_args()

DETECTOR = DETECTOR_CONVERTER[args.detector]
OBJECTIVE_NAME = args.objective
OBJECTIVE = OBJECTIVE_CONVERTER[OBJECTIVE_NAME]
MIXUP = args.objective == "mixup"

TRAIN = args.mode in ["train", "traineval"]
EVAL = args.mode in ["traineval", "eval"]

SPLIT = f"split{args.split}"
DATE = datetime.today().strftime("%Y_%m_%d")

if args.save_folder is None:
    SAVE_ROOT = os.path.join(os.curdir, "models", f"{SPLIT}-{DATE}-{OBJECTIVE_NAME}")
    # SAVE_ROOT = f"./models/{SPLIT}-{datetime.today().strftime('%Y_%m_%d')}"
    if not os.path.exists(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
else:
    SAVE_ROOT = os.path.join(os.curdir, "models", args.save_folder)

    if not os.path.exists(SAVE_ROOT):
        if EVAL and not TRAIN:
            raise RuntimeError(f"Subfolder of ./models not found: {args.save_folder}")
        os.mkdir(SAVE_ROOT)

if args.checkpoint is not None and args.save_folder is None:
    raise RuntimeError(f"Checkpoint {args.checkpoint} set without save folder.")

N_EXPERIMENTS = args.n_experiments
IM_SIZE = args.imsize

PRETRAINED = args.not_pretrained
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
SAVE_FREQ = args.save_freq
CHECKPOINT = args.checkpoint

LR = args.lr
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
GAMMA = args.gamma
learning_rate_decay_schedule = [30, 60, 90]
OOD_RATIO = 0.2

SEED = args.seed
if SEED == -1:
    SEED = random.randrange(sys.maxsize)

MIXUP_ALPHA = 2.0
MIXUP_BETA = 5.0
MIX_OPERATION = args.mix_op

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark

print(f"Using device {DEVICE}")
print(f"Process ID: {os.getpid()}")

classes_converter = {
    "split1": 79,
    "split2": 80,
    "split3": 78,
    "split4": 80,
}

N_ID_CLASSES = classes_converter[SPLIT]


def create_model():
    model = resnet50(pretrained=PRETRAINED).to(DEVICE)
    model.fc = nn.Linear(2048, N_ID_CLASSES).to(DEVICE)
    model = nn.DataParallel(model)
    # model.fc = nn.Linear(2048, N_ID_CLASSES).to(DEVICE)
    return model


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


soft_xent = SoftCE()


def eval(model, test_loader, detector=DETECTOR, limit=20):
    """Evaluate the model using the given dataloader

    Args:
        model (nn.Module): Model to Evaluate
        test_loader (Dataloader): _description_
        detector (pytorch_ood.Detector, optional): pytorch OOD detector. Defaults to DETECTOR.
        limit (int, optional): Number of batches to evaluate on. Defaults to 20.
        trainloader (_type_, optional): loader of training data, if needed for detector's `fit` method. Defaults to None.

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


def train(model, optimizer, dataloader):
    """Train the given model for one epoch

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): opimizer configured with neccessary model parameters
        dataloader (_type_): Training data to iterate through

    Returns:
        _type_: _description_
    """

    model.train()

    running_correct = 0.0
    running_total = 0.0
    running_loss_sum = 0.0

    if MIXUP:
        dataloader, oe_loader = dataloader
        oe_loader = iter(oe_loader)

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        if MIXUP:
            try:
                oe_x, _ = next(oe_loader)
            except StopIteration:
                continue
            assert (
                oe_x.shape[0] == BATCH_SIZE
            ), f"Expected Outlier batch size {BATCH_SIZE} but was actually {oe_x.shape[0]}"

            oe_x = oe_x.to(DEVICE)
            one_hot_y = torch.zeros(BATCH_SIZE, N_ID_CLASSES).to(DEVICE)
            one_hot_y.scatter_(1, y.view(-1, 1), 1)

            # ID Loss
            logits = model(X)
            id_loss = F.cross_entropy(logits, y)

            # MixOE Loss
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

            if MIX_OPERATION == "cutmix":
                mixed_x = X.clone().detach()
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                # adjust lambda to exactly match pixel ratio
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2])
                )
                # we empirically find that pasting outlier patch into ID data performs better
                # than pasting ID patch into outlier data
                mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_x[:, :, bbx1:bbx2, bby1:bby2]
            elif MIX_OPERATION == "mixup":
                mixed_x = lam * X + (1 - lam) * oe_x
            else:
                raise NotImplementedError

            oe_y = torch.ones(oe_x.size(0), N_ID_CLASSES).to(DEVICE) / N_ID_CLASSES
            mixed_y = lam * one_hot_y + (1 - lam) * oe_y

            mixed_loss = soft_xent(model(mixed_x), mixed_y)

            # OOD Loss
            oe_loss = F.cross_entropy(model(oe_x), oe_y)

            loss = id_loss + MIXUP_BETA * (mixed_loss + oe_loss)
        else:
            outputs = model(X)

            loss = OBJECTIVE(outputs, y)

            _, preds = outputs.max(dim=1)

            running_correct += preds.eq(y).sum().item()

        optimizer.zero_grad()
        model.zero_grad()

        running_total += y.size(0)

        if OBJECTIVE_NAME == "eosl":
            loss = loss.mean()

        running_loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    return running_correct / running_total * 100, running_loss_sum / running_total


def main():

    ################################################################
    #                         Loading Data                         #
    ################################################################

    DATASETS = ["ID", "aircraft_ood"]

    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Using random seed {SEED}")

    print("Datasets loaded successfully.")

    for experiment in range(1, N_EXPERIMENTS + 1):

        trainloader = training_data(SPLIT, BATCH_SIZE, IM_SIZE, MIXUP, OOD_RATIO)
        testloader = testing_data(SPLIT, BATCH_SIZE, IM_SIZE)

        ################################################################
        #                           Training                           #
        ################################################################

        if TRAIN:

            print_template = "[{}] Epoch [ {:03} / {} ] LR: {} Train Accuracy: {:3.3f}%  Epoch Loss: {:.6f}  Time:{:.3f}s"

            print("**********************************************************")
            print(
                "Starting Experiment: {} / {} with loss {}".format(
                    experiment, N_EXPERIMENTS, OBJECTIVE_NAME
                )
            )
            print("**********************************************************")

            model = create_model()

            starting_epoch = 1
            if CHECKPOINT is not None:
                model.load_state_dict(
                    torch.load(
                        os.path.join(
                            SAVE_ROOT, str(experiment), f"epoch_{CHECKPOINT}.pt"
                        )
                    )()
                )
                starting_epoch = CHECKPOINT + 1
                print(f"Epoch {CHECKPOINT} successfully loaded.")

            optimizer = torch.optim.SGD(
                model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
            )
            learning_rate_table = helpers.create_learning_rate_table(
                LR, learning_rate_decay_schedule, GAMMA, N_EPOCHS + 1
            )

            for epoch in range(starting_epoch, N_EPOCHS + 1):
                epoch_start_time = time.time()
                helpers.adjust_learning_rate(optimizer, epoch, learning_rate_table)

                train_acc, train_loss = train(model, optimizer, trainloader)

                save_this_epoch = (
                    epoch % SAVE_FREQ == 0 and SAVE_FREQ != -1
                ) or epoch == N_EPOCHS

                if save_this_epoch:
                    SAVE_PATH = os.path.join(SAVE_ROOT, str(experiment))
                    if not os.path.exists(SAVE_PATH):
                        os.mkdir(SAVE_PATH)
                    torch.save(
                        model.state_dict, os.path.join(SAVE_PATH, f"epoch_{epoch}.pt")
                    )

                metrics = eval(model, testloader)

                print_additional = " | L1 AUCROC: {:3.3f} L1 Acc@95TPR: {:3.3f} L1 FPR@95TPR: {:3.3f}".format(
                    metrics["AUROC"] * 100,
                    metrics["ACC95TPR"] * 100,
                    metrics["FPR95TPR"] * 100,
                )

                print(
                    print_template.format(
                        experiment,
                        epoch,
                        N_EPOCHS,
                        learning_rate_table[epoch - 1],
                        train_acc,
                        train_loss,
                        time.time() - epoch_start_time,
                    )
                    + print_additional
                )

        ################################################################
        #                          Evaluating                          #
        ################################################################

        if EVAL:

            if not TRAIN:

                model = create_model()

                if CHECKPOINT is not None:
                    loaded_params = os.path.join(
                        SAVE_ROOT, str(experiment), f"epoch_{CHECKPOINT}.pt"
                    )
                    assert os.path.exists(loaded_params)
                else:

                    params = glob.glob(
                        os.path.join(SAVE_ROOT, str(experiment), "epoch_*.pt")
                    )

                    def get_epoch(x):
                        return int(
                            x.split("/")[-1].replace(".pt", "").replace("epoch_", "")
                        )

                    loaded_params = sorted(params, key=get_epoch)[-1]

                model.load_state_dict(torch.load(loaded_params)())
                model.eval()

                print(f"Model loaded successfully from {loaded_params}")

            print("Evaluating.")

            obj_name_output = OBJECTIVE_NAME
            if OBJECTIVE_NAME == "mixup" and MIX_OPERATION == "cutmix":
                obj_name_output = "cutmix"

            identifiers = [
                SPLIT,
                DATE,
                experiment,
                SEED,
                obj_name_output,
                N_EPOCHS,
                BATCH_SIZE,
                OOD_RATIO,
                IM_SIZE,
            ]

            with open("./exp_results.csv", "a") as results_file:
                for detector_scheme in DETECTOR_CONVERTER.values():
                    detector_name = detector_scheme.__name__

                    indiv_results = [detector_name]

                    for level in [1, 2, 3]:

                        del testloader
                        testloader = testing_data(SPLIT, BATCH_SIZE, IM_SIZE, level)

                        results = eval(model, testloader, detector_scheme, limit=-1)

                        indiv_results += [
                            results["AUROC"],
                            results["AUPR-IN"],
                            results["AUPR-OUT"],
                            results["ACC95TPR"],
                            results["FPR95TPR"],
                        ]

                    # print(", ".join(indiv_results))
                    line = identifiers + indiv_results
                    line = [str(x) for x in line]
                    results_file.write(",".join(line) + "\n")


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt detected. Clearing Cache...")
        torch.cuda.empty_cache()

    print("Process Ending.")
