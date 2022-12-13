from argparse import ArgumentParser
from datetime import datetime
import os
import random
import torch
import torch.nn as nn

from data import combined_test_loader, training_data, classes_converter, training_planes
from evaluator import Evaluator
from lightning_models import OEModel, QuadModel, TernaryModel, VanillaModel, MixupModel

import pytorch_ood

import pytorch_lightning as pl
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DataParallelStrategy



def combined_loader(split, batch_size, ood_ratio, num_workers=None):

    # tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    # ])

    # ID_set = MNIST('/data/MNIST', download=True, transform=tf)
    # OOD_set = EMNIST('/data/EMNIST', split='letters', download=True, transform=tf)

    # loaders = {'ID': DataLoader(ID_set, batch_size=50, num_workers=36), "OOD": DataLoader(OOD_set, batch_size=150, num_workers=12)}

    if num_workers is None: num_workers = (18, 36)

    loaders = training_data(split, batch_size, 'big', ood_ratio, num_workers=num_workers)
    loaders = {"ID": loaders[0], "OOD": loaders[1]}
    return pl.trainer.supporters.CombinedLoader(loaders)


def ternary_loader(split, batch_size, ood_ratio, num_workers=None):
    
    if num_workers is None: num_workers = (16, 16, 16)

    id_loader, ood_loader = training_data(split, batch_size, 'big', ood_ratio, num_workers)
    supp_id_loader = training_planes(split, batch_size, 'big', num_workers[-1], supplement=True)

    loaders = {"ID": id_loader, "OOD": ood_loader, "SUPP_ID": supp_id_loader}

    return pl.trainer.supporters.CombinedLoader(loaders)




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
    "vanilla": None,
    "ternary": None,
    "quad": None
}



if __name__ == '__main__':

    ################################################################
    #                        CLI Arguments                         #
    ################################################################    

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

    parser.add_argument('--ternary_gamma', type=float, default=5.0)
    parser.add_argument('--quad_delta', type=float, default=5.0)

    parser.add_argument("--save_folder", "-f", type=str)
    parser.add_argument("--eval_output", type=str, default='./results.csv')

    args = parser.parse_args()

    OBJECTIVE_NAME = args.objective
    OBJECTIVE = OBJECTIVE_CONVERTER[OBJECTIVE_NAME]
    DETECTOR_NAME = args.detector
    DETECTOR = DETECTOR_CONVERTER[DETECTOR_NAME]

    MIXUP = args.objective == "mixup"
    MIX_OP = args.mix_op
    VANILLA = args.objective == "vanilla"
    TERNARY = args.objective in ["ternary", 'quad']

    SPLIT = f"split{args.split}"
    N_EXPERIMENTS = args.n_experiments
    TRAIN = args.mode in ["train", "traineval"]
    EVAL = args.mode in ["traineval", "eval"]

    DATE = DATE = datetime.today().strftime("%Y-%m-%d")

    model_dir = os.path.join(os.curdir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_split_dir = os.path.join(model_dir, SPLIT)
    if not os.path.exists(model_split_dir):
        os.mkdir(model_split_dir)

    if args.save_folder is None:
        SAVE_ROOT = os.path.join(model_split_dir, f"{DATE}-{OBJECTIVE_NAME}")
        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)
    else:
        SAVE_ROOT = os.path.join(model_dir, args.save_folder)
        if not os.path.exists(SAVE_ROOT):
            if EVAL and not TRAIN:
                raise RuntimeError(f"Subfolder of ./models not found: {args.save_folder}")
            os.mkdir(SAVE_ROOT)

    RESULTS_FILE = args.eval_output

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs

    N_ID_CLASSES = classes_converter[SPLIT]

    OOD_RATIO = args.ood_ratio
    TERNARY_GAMMA = args.ternary_gamma
    QUAD_DELTA = args.quad_delta

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
        )

        pl.utilities.seed.seed_everything(SEED)

        testing_loader = combined_test_loader(SPLIT, BATCH_SIZE, 'big', num_workers=12)

        if TERNARY:
            if args.objective == 'quad':
                model = QuadModel(N_ID_CLASSES, None, N_EPOCHS, MIX_OP, DETECTOR)
                model.set_delta(QUAD_DELTA)
            else:
                model = TernaryModel(N_ID_CLASSES, None, N_EPOCHS, MIX_OP, DETECTOR)
            model.set_ternary_gamma(TERNARY_GAMMA)
            training_loader = ternary_loader(SPLIT, BATCH_SIZE, 1, num_workers=(1,1,1))
        elif VANILLA:
            model = VanillaModel(N_ID_CLASSES, nn.CrossEntropyLoss(), N_EPOCHS, DETECTOR)
            training_loader, _ = training_data(SPLIT, BATCH_SIZE, 'big', 1, num_workers=(36, 0))
        elif MIXUP:
            model = MixupModel(N_ID_CLASSES, None, N_EPOCHS, MIX_OP, DETECTOR)
            training_loader = combined_loader(SPLIT, BATCH_SIZE, 1, num_workers=(18, 18))
        else:
            model = OEModel(N_ID_CLASSES, OBJECTIVE, N_EPOCHS, DETECTOR)
            training_loader = combined_loader(SPLIT, BATCH_SIZE, OOD_RATIO, num_workers=(18, 18))


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
                    experiment, N_EXPERIMENTS, OBJECTIVE_NAME, SPLIT[-1]
                )
            )
            print("**********************************************************")



            if TRAIN:
                start_time = datetime.now()
                trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=testing_loader)
                model.print(f"Total training time: {datetime.now() - start_time}")

            if EVAL:

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

                if OBJECTIVE_NAME == 'energy': DETECTOR = pytorch_ood.detector.NegativeEnergy

                output += Evaluator(model, DETECTOR, SPLIT, BATCH_SIZE, 'big')()
                
                with open(RESULTS_FILE, "a") as results_file:
                    line = ",".join([str(x) for x in output]) + "\n"
                    results_file.write(line)


        except KeyboardInterrupt:
            "Interrupt Detected. Attempting to shutdown gracefully"
            torch.cuda.empty_cache()
            break