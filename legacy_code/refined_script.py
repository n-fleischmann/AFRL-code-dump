import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as utilsdata
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import create_model

import pytorch_ood

from argparse import ArgumentParser
import os
from datetime import datetime
import sys
from typing import List

from fgcv_aircraft_code import Dataset_fromTxtFile as txtfiledataset
import fgcv_aircraft_code.helpers as helpers

from data import combined_test_loader, training_data, classes_converter
from trainer import StandardTrainer, MixupTrainer, VanillaTrainer
from legacy_code.save_manager import SaveManager
from evaluator import Evaluator

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


################################################################
#                    Command Line Arguments                    #
################################################################


def softmax_temp(model):
    return pytorch_ood.detector.Softmax(model, t=1000)


DETECTOR_CONVERTER = {
    "odin": pytorch_ood.detector.ODIN,
    "softmax": pytorch_ood.detector.Softmax,
    "energy": pytorch_ood.detector.NegativeEnergy,
    "softmax_temp": softmax_temp,
}

OBJECTIVE_CONVERTER = {
    "ring": pytorch_ood.loss.ObjectosphereLoss(alpha=0.0001, xi=50),
    "eosl": pytorch_ood.loss.EntropicOpenSetLoss(),
    "oe": pytorch_ood.loss.OutlierExposureLoss(),
    "energy": pytorch_ood.loss.EnergyRegularizedLoss(
        alpha=0.1, margin_in=27, margin_out=5
    ),
    "cross_entropy": pytorch_ood.loss.CrossEntropyLoss(),
    "mixup": None,
}


parser = ArgumentParser()

parser.add_argument(
    "split", default=1, type=int, help="ID/OOD Split to use. One of [1, 2, 3, 4]."
)
parser.add_argument("n_experiments", type=int, help="Number of models to train")
parser.add_argument(
    "objective",
    choices=OBJECTIVE_CONVERTER.keys(),
    help="Training Objective/Loss Function",
)
parser.add_argument("--detector", "-d", choices=DETECTOR_CONVERTER.keys(), default="softmax")
parser.add_argument(
    "--mode",
    "-m",
    default="traineval",
    choices=["train", "traineval", "eval"],
    help="Training/Evaluating mode, or combination. [Default: traineval]",
)
parser.add_argument(
    "--imsize",
    "-i",
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
parser.add_argument("--save_folder", help="Subfolder in './models/ to save/load models to/from")
parser.add_argument(
    "--checkpoint",
    type=int,
    help="Epoch checkpoint to load for training/evaluating. Will throw an error if save_folder is not specified.",
)

parser.add_argument(
    "--vanilla", 
    action="store_true",
    help="If set, train without outlier exposure (only on ID training set)",
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

DETECTOR_NAME = args.detector
DETECTOR = DETECTOR_CONVERTER[DETECTOR_NAME]
OBJECTIVE_NAME = args.objective
OBJECTIVE = OBJECTIVE_CONVERTER[OBJECTIVE_NAME]
MIXUP = args.objective == "mixup"
VANILLA = args.vanilla

TRAIN = args.mode in ["train", "traineval"]
EVAL = args.mode in ["traineval", "eval"]

SPLIT = f"split{args.split}"
DATE = datetime.today().strftime("%Y_%m_%d")

# Make sure the general model save dir exists
model_dir = os.path.join(os.curdir, "models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Make sure the specific save folder exists
if args.save_folder is None:
    SAVE_ROOT = os.path.join(model_dir, f"{SPLIT}-{DATE}-{OBJECTIVE_NAME}")
    # SAVE_ROOT = f"./models/{SPLIT}-{datetime.today().strftime('%Y_%m_%d')}"
    if not os.path.exists(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
else:
    SAVE_ROOT = os.path.join(os.curdir, "models", args.save_folder)

    if not os.path.exists(SAVE_ROOT):
        if EVAL and not TRAIN:
            raise RuntimeError(f"Subfolder of ./models not found: {args.save_folder}")
        os.mkdir(SAVE_ROOT)

# Throw an error if checkpoint is specified without the save_folder
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
OOD_RATIO = 3

SEED = args.seed
if SEED == -1:
    SEED = random.randrange(sys.maxsize)

MIX_OPERATION = args.mix_op

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

print(f"Using device {DEVICE}")
print(f"Process ID: {os.getpid()}")


N_ID_CLASSES = classes_converter[SPLIT]


def main() -> None:

    ################################################################
    #                         Loading Data                         #
    ################################################################

    DATASETS = ["ID", "aircraft_ood"]

    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Using random seed {SEED}")

    print("Datasets loaded successfully.")

    for experiment in range(1, N_EXPERIMENTS + 1):

        train_loader_fn = lambda: training_data(
            SPLIT, BATCH_SIZE, IM_SIZE, OOD_RATIO
        )

        test_loader_fn = lambda: combined_test_loader(SPLIT, BATCH_SIZE, IM_SIZE)


        ################################################################
        #                           Training                           #
        ################################################################

        if TRAIN:

            print("**********************************************************")
            print(
                "Starting Experiment: {} / {} with {} and {}".format(
                    experiment, N_EXPERIMENTS, OBJECTIVE_NAME, DETECTOR_NAME
                )
            )
            print("**********************************************************")

            model = create_model(PRETRAINED, N_ID_CLASSES)
            save_manager = SaveManager(SAVE_ROOT, SAVE_FREQ, N_EPOCHS, experiment)

            starting_epoch = 1
            if CHECKPOINT is not None:
                model, starting_epoch = save_manager.load(model, CHECKPOINT)

            if MIXUP:
                trainer = MixupTrainer(
                    model,
                    OBJECTIVE,
                    N_EPOCHS,
                    SPLIT,
                    MIX_OPERATION,
                    starting_epoch=CHECKPOINT,
                )
            elif VANILLA:
                trainer = VanillaTrainer(
                    model, OBJECTIVE, N_EPOCHS, starting_epoch=starting_epoch
                )
            else:
                trainer = StandardTrainer(
                    model, OBJECTIVE, N_EPOCHS, starting_epoch=starting_epoch
                )

            trainer.initialize_optimizer(LR, MOMENTUM, WEIGHT_DECAY, GAMMA)

            model = trainer.train(train_loader_fn, test_loader_fn, save_manager)

        ################################################################
        #                          Evaluating                          #
        ################################################################

        if EVAL:

            if not TRAIN:

                model = create_model(PRETRAINED, N_ID_CLASSES)
                save_manager = SaveManager(SAVE_ROOT, SAVE_FREQ, N_EPOCHS, experiment)

                if CHECKPOINT is not None:
                    model, starting_epoch = save_manager.load(model, CHECKPOINT)
                else:
                    model, starting_epoch = save_manager.find_and_load(model)

                model.eval()

            print("Evaluating.")

            obj_name_output = OBJECTIVE_NAME
            if OBJECTIVE_NAME == "mixup" and MIX_OPERATION == "cutmix":
                obj_name_output = "cutmix"

            output = [
                SPLIT,
                DATE,
                experiment,
                SEED,
                obj_name_output,
                N_EPOCHS,
                BATCH_SIZE,
                IM_SIZE,
                DETECTOR.__name__,
            ]

            output += Evaluator(model, DETECTOR, SPLIT, BATCH_SIZE, IM_SIZE)()

            with open("./results.csv", "a") as results_file:

            #     all_results = {}
            #     levels = [1, 2, 3, -1]
            #     for level in levels:

            #         testloader = testing_data(SPLIT, BATCH_SIZE, IM_SIZE, level)

            #         all_results[level] = eval(model, testloader, DETECTOR, limit=-1)

            #     eval_results = [ID_accuracy(model, testloader, -1)]
            #     eval_results += [all_results[level]["AUROC"] for level in levels]
            #     eval_results += [all_results[level]["FPR95TPR"] for level in levels]

            #     output += eval_results
                line = ",".join([str(x) for x in output]) + "\n"
                results_file.write(line)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt detected. Clearing Cache...")
        torch.cuda.empty_cache()

    print("Process Ending.")
