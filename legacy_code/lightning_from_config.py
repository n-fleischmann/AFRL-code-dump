from argparse import ArgumentParser
from datetime import datetime
import json
import os
import random
from re import M
from typing import Dict
import torch
import torch.nn as nn

from data import combined_loader, combined_test_loader, ternary_loader, training_data, classes_converter, training_planes
from evaluator import Evaluator
from lightning_models_dict import OEModel, QuadModel, TernaryModel, VanillaModel, MixupModel

import pytorch_ood

import pytorch_lightning as pl
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DataParallelStrategy


def softmax_temp(model: nn.Module) -> pytorch_ood.detector:
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


OBJECTIVE_TO_MODEL_CONVERTER = {
    "oe": OEModel,
    "energy": OEModel,
    "mixup": MixupModel,
    "ternary": TernaryModel,
    "quad": QuadModel,
}

def parse_config(path: os.PathLike) -> Dict[str, object]:
    
    assert os.path.exists(path), f"Path does not exist: {path}"
    assert os.path.splitext(path)[-1] in ['.config', '.json'], f"Config must be filetype *.config or *.json, got {path}"

    with open(path, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    config['n_id_classes'] = classes_converter[config['split']] # dictionary converts from split name -> # of id classes
    config['objective'] = OBJECTIVE_CONVERTER[config['objective_name']] # loss fn name -> loss fn implementation
    config['detector'] = DETECTOR_CONVERTER[config['detector_name']] # detector name -> detector implementation
    
    return config


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "config", os.PathLike,
        help="json config file (either *.config or *.json) of hyperparameters"
    )

    parser.add_argument(
        "mode",
        default="traineval",
        choices=["train", "traineval", "eval"],
        help="Training/Evaluating mode, or combination. [Default: traineval]",
    )

    args = parser.parse_args()

    config = parse_config(args.config)

    TRAIN = args.mode in ["train", "traineval"]
    EVAL = args.mode in ["traineval", "eval"]

    SAVE_ROOT = os.curdir
    for dir in ["models", config['split'], config['model_dir'], config['experiment']]:
        SAVE_ROOT = os.path.join(SAVE_ROOT, dir)
        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

    SEED = config.get('seed', -1)
    if SEED == -1:
        SEED = random.randrange(4294967295)
    
    pl.utilities.seed.seed_everything(SEED)

    model = OBJECTIVE_TO_MODEL_CONVERTER[config["objective_name"]](config)

    training_loader = model.get_training_loader()
    testing_loader = combined_test_loader(config['split'], config['batch_size'], 'big', num_workers=12)


    try:
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(SAVE_ROOT),
            filename="{epoch}",
            every_n_epochs=20,
            save_top_k=-1
        )

        trainer = pl.Trainer(
                strategy=DataParallelStrategy(),
                accelerator='gpu', 
                devices=-1, 
                enable_progress_bar=False, 
                max_epochs=config['max_epochs'],
                callbacks=[checkpoint_callback]
        )


        print(f"Using PID: {os.getpid()}")
        print("**********************************************************")
        print(
            "Starting Experiment {} with {} on split {}".format(
                config['experiment'], config['objective_name'], config['split'][-1]
            )
        )
        print("**********************************************************")

        if TRAIN:
            start_time = datetime.name()
            trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=testing_loader)
            model.print(f"Total training time: {datetime.now() - start_time}")
        
        if EVAL:

            if not TRAIN:
                model.load_state_dict(
                    torch.load(
                        os.path.join(SAVE_ROOT, 'epoch=99.ckpt')
                    )['state_dict']
                )
        
            model.eval()

            obj_name_output = OBJECTIVE_NAME
            if OBJECTIVE_NAME == "mixup" and MIX_OP == "cutmix":
                obj_name_output = "cutmix"














if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:

        print("Keyboard Interrupt Detected. Shutting down gracefully...")
        torch.cuda.empty_cache()

    print("Process Ended.")
