from argparse import ArgumentParser
from asyncio.constants import ACCEPT_RETRY_DELAY
from datetime import datetime
import os
import random
from venv import create
import torch
import torch.nn as nn

from data import combined_test_loader, get_weights, planes_classes_converter, ships_classes_converter, dataset_converter, testing_id
from evaluator import Evaluator
from lightning_models_dict import OEModel, QuadModel, TernaryModel, VanillaModel, MixupModel

import pytorch_ood

from torchmetrics import Accuracy
import pytorch_lightning as pl
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DataParallelStrategy




def softmax_temp(model):
    return pytorch_ood.detector.Softmax(model, t=1000)


DETECTOR_CONVERTER = {
    "odin": pytorch_ood.detector.ODIN,
    "softmax": pytorch_ood.detector.Softmax,
    "energy": pytorch_ood.detector.NegativeEnergy,
    "softmax_temp": softmax_temp,
}


def energy_with_margin(margin: float):
    return pytorch_ood.loss.EnergyRegularizedLoss(
        alpha=0.1, margin_in=margin, margin_out=5
    )


OBJECTIVE_CONVERTER = {
    "ring": pytorch_ood.loss.ObjectosphereLoss(alpha=0.0001, xi=50),
    "eosl": pytorch_ood.loss.EntropicOpenSetLoss(),
    "oe": pytorch_ood.loss.OutlierExposureLoss(),
    "energy": energy_with_margin,
    "cross_entropy": pytorch_ood.loss.CrossEntropyLoss(),
    "mixup": None,
    "vanilla": torch.nn.CrossEntropyLoss(),
    "ternary": None,
    "quad": None
}


OBJECTIVE_TO_MODEL_CONVERTER = {
    "oe": OEModel,
    "energy": OEModel,
    "vanilla": VanillaModel,
    "mixup": MixupModel,
    "ternary": TernaryModel,
    "quad": QuadModel,
}

def create_if_dir_does_not_exist_recursive(dir):

    if os.path.exists(dir):
        print(f"Dir @ {dir} already exists")
        return

    working_path = ''

    for item in dir.split(os.path.sep):
        if item == '': item = os.path.sep
        working_path = os.path.join(working_path, item)
        if not os.path.exists(working_path):
            os.mkdir(working_path)

    print(f"Created dir @ {working_path}")


if __name__ == '__main__':

    ################################################################
    #                        CLI Arguments                         #
    ################################################################    

    parser = ArgumentParser()

    parser.add_argument(
        "split", 
        help="ID/OOD Split to use. One of [1, 2, 3, 4] for planes or ['military', 'civ', 'hard1', 'hard2'] for ships.",
        choices = list(planes_classes_converter.keys()) + list(ships_classes_converter.keys())
    )

    parser.add_argument("n_experiments", type=int, help="Number of models to train")
    parser.add_argument(
        "objective",
        choices=OBJECTIVE_CONVERTER.keys(),
        help="Training Objective/Loss Function",
    )
    parser.add_argument("--detector", "-d", choices=DETECTOR_CONVERTER.keys(), default="softmax")

    parser.add_argument("--dataset", choices=['planes', 'ships'], default='planes')

    parser.add_argument(
        "--seed",  "-s",
        type=int,
        default=-1,
        help="Random seed for model initialization and random shuffling.",
    )

    parser.add_argument(
        "--mode",
        "-m",
        default="traineval",
        choices=["train", "traineval", "eval"],
        help="Training/Evaluating mode, or combination. [Default: traineval]",
    )

    parser.add_argument("--batch_size", "-b", type=int, default=20)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--mix_op", choices=["mixup", "cutmix"], default="cutmix")
    parser.add_argument("--ood_ratio", "-o", type=int, default=3)

    parser.add_argument('--mixup_beta', type=float, default=5.0)
    parser.add_argument('--ternary_gamma', type=float, default=5.0)
    parser.add_argument('--quad_delta', type=float, default=5.0)
    parser.add_argument('--energy_margin_in', type=float, default=-27.0)

    parser.add_argument("--save_folder", "-f", type=str)
    parser.add_argument("--eval_output", type=str, default='./results.csv')
    parser.add_argument("--weighted", action='store_true')

    args = parser.parse_args()


    OBJECTIVE_NAME = args.objective
    OBJECTIVE = OBJECTIVE_CONVERTER[OBJECTIVE_NAME]

    DETECTOR_NAME = args.detector

    if OBJECTIVE_NAME == 'energy':
        ENERGY_MARGIN_IN = args.energy_margin_in
        OBJECTIVE = OBJECTIVE(ENERGY_MARGIN_IN)
        DETECTOR_NAME = 'energy'


    DETECTOR = DETECTOR_CONVERTER[DETECTOR_NAME]
    DATASET = args.dataset

    MIXUP = args.objective == "mixup"
    MIX_OP = args.mix_op
    VANILLA = args.objective == "vanilla"
    TERNARY = args.objective == "ternary"
    QUAD = args.objective == "quad"

    assert args.split in dataset_converter[DATASET].keys(),  f"Split '{args.split}' not available for dataset {DATASET}"
    SPLIT = args.split
    N_ID_CLASSES = dataset_converter[DATASET][SPLIT]


    # if DATASET == 'planes':
    #     assert args.split in planes_classes_converter.keys()
    #     SPLIT = args.split
    #     N_ID_CLASSES = planes_classes_converter[SPLIT]
    # elif DATASET == 'ships':
    #     assert args.split in ships_classes_converter.keys()
    #     SPLIT = args.split
    #     N_ID_CLASSES = ships_classes_converter[SPLIT]
    
    N_EXPERIMENTS = args.n_experiments
    TRAIN = args.mode in ["train", "traineval"]
    EVAL = args.mode in ["traineval", "eval"]

    DATE = DATE = datetime.today().strftime("%Y-%m-%d")

    model_dir = os.path.join("/data/model_backups/models", DATASET, SPLIT)

    if args.save_folder is None:
        SAVE_ROOT = os.path.join(model_dir, f"{DATE}-{OBJECTIVE_NAME}")
    else:
        SAVE_ROOT = os.path.join(model_dir, args.save_folder)
        if not os.path.exists(SAVE_ROOT):
            if EVAL and not TRAIN:
                raise RuntimeError(f"Subfolder of ./models not found: {args.save_folder}")
    
    create_if_dir_does_not_exist_recursive(SAVE_ROOT)

    RESULTS_FILE = args.eval_output

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs

    OOD_RATIO = args.ood_ratio

    MIXUP_BETA = args.mixup_beta
    TERNARY_GAMMA = args.ternary_gamma
    QUAD_DELTA = args.quad_delta

    if args.weighted:
        OBJECTIVE.weight = get_weights(DATASET, SPLIT)

    ################################################################
    #                           Training                           #
    ################################################################

    for experiment in range(1, N_EXPERIMENTS + 1):

        SEED = args.seed
        if SEED == -1:
            SEED = random.randrange(4294967295)
        else:
            SEED += experiment # Make sure each experiment has a different but deterministic seed

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(SAVE_ROOT, str(experiment)),
            filename="{epoch}",
            every_n_epochs=20,
            save_top_k=-1
        )

        pl.utilities.seed.seed_everything(SEED)

        testing_loader = combined_test_loader(DATASET, SPLIT, BATCH_SIZE, 'big', num_workers=12)

        config = {
            "dataset": DATASET,
            "split": SPLIT,
            'batch_size': BATCH_SIZE,
            "n_id_classes": N_ID_CLASSES,
            "objective": OBJECTIVE,
            "max_epochs": N_EPOCHS,
            "ood_ratio": OOD_RATIO,
            "mixup_mode": MIX_OP,
            "mixup_beta": MIXUP_BETA,
            "ternary_gamma": TERNARY_GAMMA,
            "quad_delta": QUAD_DELTA,
        }

        model = OBJECTIVE_TO_MODEL_CONVERTER[OBJECTIVE_NAME](config)
        training_loader = model.get_training_loader()

        try:
            trainer = pl.Trainer(
                #strategy=DDPStrategy(find_unused_parameters=False), 
                strategy=DataParallelStrategy(),
                accelerator='gpu', 
                devices=-1, 
                enable_progress_bar=False, 
                max_epochs=N_EPOCHS,
                callbacks=[checkpoint_callback]
            )

            print(f"Using PID: {os.getpid()}")
            print("**********************************************************")
            print(
                "Starting Experiment: {} / {} with {} on split {}".format(
                    experiment, N_EXPERIMENTS, OBJECTIVE_NAME, SPLIT
                )
            )
            print("**********************************************************")



            if TRAIN:
                start_time = datetime.now()
                trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=testing_loader)
                model.print(f"Total training time: {datetime.now() - start_time}")

            if EVAL:

                if SPLIT == "non_other":
                    model.eval()
                    model.to("cuda:1")
                    testing_data = testing_id(DATASET, SPLIT, BATCH_SIZE, 'big')
                    accuracy = Accuracy().to("cuda:1")

                    for X, y in testing_data:
                        X, y = X.to('cuda:1'), y.to('cuda:1')
                        pred = model(X).argmax(dim=1)
                        accuracy.update(pred, y)

                    acc_value = accuracy.compute()
                    
                    with open(RESULTS_FILE, 'a') as results_file:
                        results_file.write(f"{acc_value}\n")

                else:           

                    if not TRAIN:
                        model.load_state_dict(
                            torch.load(
                                os.path.join(SAVE_ROOT, str(experiment), 'epoch=99.ckpt')
                            )['state_dict']
                        )

                    model.eval()

                    obj_name_output = OBJECTIVE_NAME
                    if OBJECTIVE_NAME == "mixup" and MIX_OP == "cutmix":
                        obj_name_output = "cutmix"

                    output = [
                        SPLIT,
                        DATE,
                        experiment,
                        SEED,
                        obj_name_output,
                        N_EPOCHS,
                        BATCH_SIZE,
                        'big',
                        DETECTOR.__name__,
                    ]

                    output += Evaluator(model, DETECTOR, DATASET, SPLIT, BATCH_SIZE, 'big')()
                    
                    with open(RESULTS_FILE, "a") as results_file:
                        line = ",".join([str(x) for x in output]) + "\n"
                        results_file.write(line)


        except KeyboardInterrupt:
            "Interrupt Detected. Attempting to shutdown gracefully"
            torch.cuda.empty_cache()
            break