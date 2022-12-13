import torch
import torch.utils.data as utilsdata
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.datasets import ImageFolder

import pytorch_ood
import os
import numpy as np
from typing import Iterator, Optional, Tuple, Union

import pytorch_lightning as pl

from fgcv_aircraft_code import Dataset_fromTxtFile as txtfiledataset


planes_classes_converter = {
    "split1": 79,
    "split2": 80,
    "split3": 78,
    "split4": 80,
}

ships_classes_converter = {
    "military": 9,
    "civ": 10,
    "hard1": 15,
    "hard2": 15,
    "non_other": 21
}

dataset_converter = {
    "planes": planes_classes_converter,
    "ships": ships_classes_converter
}

def dataset_root(dataset: str, split: str, supplement: Optional[bool]=False) -> os.PathLike:
    """Get the dataset root for the aircraft/ships data for the current split

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): one of [
            'split1', 'split2', 'split3', 'split4', 
            'military', 'civ', 'hard1', 'hard2', 'non_other'
        ]
        supplement (Optional[bool]) set to true to use the dataset copy for cutmix methods

    Returns:
        os.PathLike: path to the dataset
    """

    if dataset == 'planes':
        assert split in planes_classes_converter.keys(), \
            f"Split {split} not recognized for planes data, should be one of {planes_classes_converter.keys()}"

        if not supplement:
            path = f"/data/fgvc_aircraft/fgvc-aircraft-2013b/data/splits/{split}"
        else:
            path = f"/data/fgvc_aircraft/fgvc-aircraft-2013b/data_supplement/splits/{split}"

    elif dataset == 'ships':
        assert split in ships_classes_converter.keys(), \
            f"Split {split} not recognized for ships data, should be one of {ships_classes_converter.keys()}"
        
        if not supplement:
            path = f"/data/ships/nate/ShipRSImageNet/VOC_Format/splits/{split}"
        else:
            path = f"/data/ships/nate/ShipRSImageNet/VOC_supp/splits/{split}"
    else:
        raise NotImplementedError(f"Unrecognized dataset: {dataset}")

    assert os.path.exists(path), f"path {path} not found"
    return path


def get_weights(dataset: str, split: str) -> torch.Tensor:
    """Get class weights for classes in given dataset and split

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): one of [
            'split1', 'split2', 'split3', 'split4', 
            'military', 'civ', 'hard1', 'hard2', 'non_other'
        ]

    Returns:
        torch.Tensor: _description_
    """    

    DATASET_ROOT = dataset_root(dataset, split)

    path = os.path.join(DATASET_ROOT, 'ID_weights.txt')
    assert os.path.exists(path), f"File not found: {path}"

    with open(path, 'r') as infile:
        out_tensor = torch.ones(size=(dataset_converter[dataset][split],))
        for line in infile.readlines():
            idx, val = line.strip().split()
            out_tensor[int(idx)] = float(val)

    return out_tensor





################################################################
#                          Transforms                          #
################################################################

def get_transforms(im_size: str, train: bool = True) -> transforms.Compose:
    """Create ImageNet normalized image transforms

    Args:
        im_size (str): 'big' for 448x448 images, 'medium' for 224x224 images
        train (bool, optional): if true, return training transforms. Otherwise, return testing transforms. Defaults to True.

    Raises:
        NotImplementedError: If IM_SIZE is not big or medium

    Returns:
        transforms.Compose: composition of transforms
    """

    # ImageNet Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    if im_size == "big":

        transform_train = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomCrop((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    elif im_size == "medium":

        transform_train = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # JYZ
                transforms.RandomCrop((224, 224)),  # JYZ
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    else:
        raise NotImplementedError(
            f"Image processing for size {im_size} is not available"
        )

    return transform_train if train else transform_test


################################################################
#                  Testing and Training Data                   #
################################################################


def make_infinite(iterator: Iterator[object]):
    """Make an infinite sequential generator out of an iterator without copying data into a secondary list

    Args:
        iterator (Iterator[object]): an iterable object to make infinite

    Yields:
        object: items in the iterator
    """    
    while True:
        for item in iterator:
            yield item


def training_id(
    dataset:str, split: str, batch_size: int, im_size: str, num_workers: Optional[int] = 2, supplement: Optional[bool]=False,
) -> DataLoader:
    """Get training in-distribution images of the given dataset/split

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): split of the given dataset to fetch
        batch_size (int): number of items in each batch
        im_size (str): 'big' for 448x448 images, 'medium' for 224x224
        num_workers (Optional[int], optional): number of CPU workers in the dataloader. Defaults to 2.
        supplement (Optional[bool], optional): use the supplementary dataset for cutmix methods. Defaults to False.

    Returns:
        DataLoader: dataloader of the given dataset/split
    """

    transform_train = get_transforms(im_size)
    DATASET_ROOT = dataset_root(dataset, split, supplement=supplement)

    ID_train_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_trainset.txt", transform_train, is_fgcvaircraft=dataset=='planes'
    )

    ID_loader = DataLoader(
        ID_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # timeout=1000000
    )

    return ID_loader



def infinite_imagenet(
    batch_size: int, im_size: str, num_workers: Optional[int] = 2
) -> DataLoader:
    """Make an infinite ImageNet dataloader

    Args:
        batch_size (int): number of items in each batch
        im_size (str): 'big' for 448x448 images, 'medium' for 224x224
        num_workers (Optional[int], optional): number of CPU workers in the dataloader. Defaults to 2.

    Returns:
        DataLoader: Infinite imagenet dataloader
    """

    transform_train = get_transforms(im_size)

    OOD_train_dataset = ImageFolder(
        root="/data/imagenet/ILSVRC/Data/CLS-LOC/train/",
        transform=transform_train,
        target_transform=pytorch_ood.utils.ToUnknown(),
    )

    OOD_loader = make_infinite( # Make infinite generator
    DataLoader(
        OOD_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        # timeout=1000000
    )
)

    return OOD_loader





def training_data(
    dataset: str, split: str, batch_size: int, im_size: str, ood_ratio: int, num_workers: Optional[Tuple[int, int]]=None
) -> Tuple[DataLoader, DataLoader]:
    """create training ID/OOD dataloaders

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): the split to create dataloader for
        batch_size (int): Batch size for data
        im_size (str): image size for transforms
        ood_ratio (int): size of OOD dataset batches as multiples of ID batch_size
        num_workers (Optional[Tuple[int, int]], optional): Number of CPU workers for the ID and OOD DataLoaders respectively. Defaults to (18, 36).

    Returns:
        Tuple[DataLoader, DataLoader]: (training_id, imagenet)
    """    

    if num_workers is None: num_workers = (18, 36)

    return training_id(dataset, split, batch_size, im_size, num_workers[0]), infinite_imagenet(batch_size * ood_ratio, im_size, num_workers[1])





def testing_ood(dataset, split, batch_size, im_size, level, dataset_only=False, set_unknown=True) -> Union[Dataset, DataLoader]:
    """Make testing OOD dataloader

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): the split of the dataset to fetch from
        batch_size (int): Batch size for data
        im_size (str): image size for transforms
        level (_type_): the level of fine-grained OOD [1, 2, 3, -1] for planes, [1, 2, -1] for ships
        dataset_only (bool, optional): If true return just the dataset, not wrapped in the dataloader. Defaults to False.
        set_unknown (bool, optional): If true, set the targets to unknown using pytorch_ood, if false set targets to 1. Defaults to True.

    Returns:
        Union[Dataset, DataLoader]: your data, your way
    """    

    transform_test = get_transforms(im_size, False)
    
    DATASET_ROOT = dataset_root(dataset, split)

    if set_unknown:
        target_transform = pytorch_ood.utils.ToUnknown()
    else:
        def target_transform(x):
            return 1

    OOD_test_dataset = txtfiledataset.Dataset_fromTxtFile_with_levels(
        DATASET_ROOT + "/OOD_testset.txt",
        transform_test,
        target_transform=target_transform,
        is_fgcvaircraft=dataset=='planes',
    )
    OOD_test_dataset.set_level(level)

    if dataset_only: return OOD_test_dataset

    return DataLoader(
        OOD_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        # timeout=1000000
    )



def testing_id(dataset, split, batch_size, im_size, dataset_only=False) -> Union[Dataset, DataLoader]:
    """create the ID testing dataset/dataloader

    Args:

        dataset (str): one of ['planes', 'ships']
        split (str): split of the given dataset to fetch
        batch_size (int): number of items in each batch
        im_size (str): 'big' for 448x448 images, 'medium' for 224x224
        dataset_only (bool, optional): If true return only the dataset, if false return wrapped in a DataLoader. Defaults to False.

    Returns:
        Union[Dataset, DataLoader]: A dataset/dataloader of the given dataset and split
    """    

    transform_test = get_transforms(im_size, False)
    DATASET_ROOT = dataset_root(dataset, split)

    ID_test_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_testset.txt", transform_test, is_fgcvaircraft=dataset=='planes'
    )

    if dataset_only: return ID_test_dataset

    return DataLoader(
        ID_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        # timeout=1000000
    )


def combined_test_loader(
    dataset: str, split: str, batch_size: int, im_size: str, level: int = 1, num_workers:int=2
) -> DataLoader:
    """combined ID/OOD testing loader

    Args:
        dataset (str): one of ['planes', 'ships']
        split (str): split for the given dataset
        batch_size (int): number of samples in the batch
        im_size (str): 'big' for 448x448 images, 'medium' for 224x224
        level (int, optional): OOD level to fetch. One of [1, 2, 3, -1] for planes, [1, 2, -1] for ships. Defaults to 1.
        num_workers (int, optional): number of CPU workers in the combined dataloader. Defaults to 2.

    Returns:
        DataLoader: Combined testing ID/OOD dataloader
    """


    ID_test_dataset = testing_id(dataset, split, batch_size, im_size, dataset_only=True)

    OOD_test_dataset = testing_ood(dataset, split, batch_size, im_size, level, dataset_only=True)

    return DataLoader(
        utilsdata.ConcatDataset([ID_test_dataset + OOD_test_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        # timeout=1000
    )





################################################################
#                          Course OOD                          #
################################################################


def get_test_from_folder(path, im_size, batch_size, dataset_only=False):

    transform_test = get_transforms(im_size, False)

    dataset = ImageFolder(path, transform=transform_test, target_transform=pytorch_ood.utils.ToUnknown())

    if dataset_only: return dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        # timeout=1000
    )



def dogs(im_size, batch_size, dataset_only=False):
    return get_test_from_folder('/data/stanford_dogs/Images', im_size, batch_size, dataset_only)

def places(im_size, batch_size, dataset_only=False):
    return get_test_from_folder('/data/places/val', im_size, batch_size, dataset_only)

def cars(im_size, batch_size, dataset_only= False):
    return get_test_from_folder('/data/cars/cars_test_parent', im_size, batch_size, dataset_only)



def course_ood(im_size, batch_size, datasets_only = False):

    datasets = [dogs(im_size, batch_size, True), places(im_size, batch_size, True), cars(im_size, batch_size, True)]
    size = min([len(dataset) for dataset in datasets])

    samples = [utilsdata.random_split(dataset, [size, len(dataset) - size])[0] for dataset in datasets]

    if datasets_only: return samples

    return DataLoader(
        utilsdata.ConcatDataset(samples),
        batch_size = batch_size,
        shuffle=True,
        num_workers=2,
        # timeout=1000
    )


def dogs_places_cars(im_size, batch_size):

    datasets = course_ood(im_size, batch_size, datasets_only=True)

    return [DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        # timeout=1000
    ) for dataset in datasets]


################################################################
#                      Lightning Loaders                       #
################################################################


def combined_loader(dataset, split, batch_size, ood_ratio, num_workers=None):

    if num_workers is None: num_workers = (18, 36)

    loaders = training_data(dataset, split, batch_size, 'big', ood_ratio, num_workers=num_workers)
    loaders = {"ID": loaders[0], "OOD": loaders[1]}
    return pl.trainer.supporters.CombinedLoader(loaders)



def ternary_loader(dataset, split, batch_size, ood_ratio, num_workers=None):
    
    if num_workers is None: num_workers = (16, 16, 16)

    id_loader, ood_loader = training_data(dataset, split, batch_size, 'big', ood_ratio, num_workers)
    supp_id_loader = training_id(dataset, split, batch_size, 'big', num_workers[-1], supplement=True)

    loaders = {"ID": id_loader, "OOD": ood_loader, "SUPP_ID": supp_id_loader}

    return pl.trainer.supporters.CombinedLoader(loaders)




################################################################
#                       Distributed Data                       #
################################################################


# def distributed_training_data(
#     split: str, batch_size: int, im_size: str, ood_ratio: int, rank: int, world_size:int
# ) -> Tuple[DataLoader, DataLoader]:
    
#     transform_train = get_transforms(im_size)

#     DATASET_ROOT = dataset_root(split)

#     ID_train_dataset = txtfiledataset.Dataset_fromTxtFile(
#         DATASET_ROOT + "/ID_trainset.txt", transform_train, is_fgcvaircraft=dataset=='planes'
#     )

#     OOD_train_dataset = ImageFolder(
#         root="/data/imagenet/ILSVRC/Data/CLS-LOC/train/",
#         transform=transform_train,
#         target_transform=pytorch_ood.utils.ToUnknown(),
#     )

#     ID_sampler = utilsdata.DistributedSampler(
#         ID_train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
#     )

#     OOD_sampler = utilsdata.DistributedSampler(
#         OOD_train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
#     )

#     ID_loader = DataLoader(
#         ID_train_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=False,
#         sampler=ID_sampler,
#         # timeout=1000
#     )

#     OOD_loader = make_infinite( # Make infinite generator
#         DataLoader(
#             OOD_train_dataset,
#             batch_size=ood_ratio * batch_size,
#             shuffle=False,
#             num_workers=0,
#             drop_last=True,
#             pin_memory=False,
#             sampler=OOD_sampler,
#             # timeout=1000
#         )
#     )

#     return ID_loader, OOD_loader
