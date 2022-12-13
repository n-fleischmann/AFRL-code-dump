import torch.utils.data as utilsdata
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

import pytorch_ood
import os
import numpy as np
from typing import Optional, Tuple

from fgcv_aircraft_code import Dataset_fromTxtFile as txtfiledataset


classes_converter = {
    "split1": 79,
    "split2": 80,
    "split3": 78,
    "split4": 80,
}

def dataset_root(split: str, supplement: Optional[bool]=False) -> os.PathLike:
    """Get the dataset root for the aircraft data for the current split

    Args:
        split (str): one of ['split1', 'split2', 'split3', 'split4']

    Returns:
        os.PathLike: path to the dataset
    """
    assert split in [
        f"split{n}" for n in range(1, 5)
    ], f"split '{split}' not recognized!"
    if not supplement:
        path = f"/data/fgvc_aircraft/fgvc-aircraft-2013b/data/splits/{split}"
    else:
        path = f"/data/fgvc_aircraft/fgvc-aircraft-2013b/data_supplement/splits/{split}"
    assert os.path.exists(path), f"path {path} not found"
    return path


################################################################
#                          Transforms                          #
################################################################

def get_transforms(im_size: str, train: bool = True) -> transforms.Compose:
    """Create ImageNet normalized image transforms

    Args:
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


def make_infinite(iterator):

    while True:
        for item in iterator:
            yield item

def training_planes(
    split: str, batch_size: int, im_size: str, num_workers: Optional[int] = 2, supplement: Optional[bool]=False
) -> DataLoader:

    transform_train = get_transforms(im_size)
    DATASET_ROOT = dataset_root(split, supplement=supplement)

    ID_train_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_trainset.txt", transform_train, is_fgcvaircraft=True
    )

    ID_loader = DataLoader(
        ID_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        timeout=1000
    )
 

    return ID_loader

def infinite_imagenet(
    split: str, batch_size: int, im_size: str, num_workers: Optional[int] = 2
) -> DataLoader:

    transform_train = get_transforms(im_size)
    DATASET_ROOT = dataset_root(split)

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
        timeout=1000
    )
)

    return OOD_loader





def training_data(
    split: str, batch_size: int, im_size: str, ood_ratio: int, num_workers: Optional[Tuple[int, int]]=None
) -> Tuple[DataLoader, DataLoader]:
    """Fetch Training data

    Args:
        split (str): one of ['split1', 'split2', 'split3', 'split4']
        batch_size (int): Batch size for data
        im_size (str): image size for transforms
        ood_ratio (int): size of OOD dataset batches as multiples of ID batch_size
        num_workers (Optional[Tuple[int, int]], optional): Number of CPU workers for the ID and OOD DataLoaders respectively. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: _description_
    """    

    if num_workers is None: num_workers = (18, 36)

    return training_planes(split, batch_size, im_size, num_workers[0]), infinite_imagenet(split, batch_size * ood_ratio, im_size, num_workers[1])





def testing_ood_planes(split, batch_size, im_size, level, dataset_only=False, set_unknown=True):

    transform_test = get_transforms(im_size, False)
    
    DATASET_ROOT = dataset_root(split)

    if set_unknown:
        target_transform = pytorch_ood.utils.ToUnknown()
    else:
        def target_transform(x):
            return 1

    OOD_test_dataset = txtfiledataset.Dataset_fromTxtFile_with_levels(
        DATASET_ROOT + "/OOD_testset.txt",
        transform_test,
        target_transform=target_transform,
        is_fgcvaircraft=True,
    )
    OOD_test_dataset.set_level(level)

    if dataset_only: return OOD_test_dataset

    return DataLoader(
        OOD_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        timeout=1000
    )



def testing_id_planes(split, batch_size, im_size, dataset_only=False):

    transform_test = get_transforms(im_size, False)
    DATASET_ROOT = dataset_root(split)

    ID_test_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_testset.txt", transform_test, is_fgcvaircraft=True
    )

    if dataset_only: return ID_test_dataset

    return DataLoader(
        ID_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        timeout=1000
    )


def combined_test_loader(
    split: str, batch_size: int, im_size: str, level: int = 1, num_workers:int=2
) -> DataLoader:
    """Get the testing data

    Args:
        split (str):  one of ['split1', 'split2', 'split3', 'split4']
        batch_size (int): Batch Size
        im_size (str): Image Size
        level (int, optional): OOD classification Level, one of [1, 2, 3, -1]. Defaults to 1.
             1 for manufacturer
             2 for variant
             3 for model
            -1 for all.

    Returns:
        DataLoader: Dataloader of loaded data
    """

    ID_test_dataset = testing_id_planes(split, batch_size, im_size, dataset_only=True)

    OOD_test_dataset = testing_ood_planes(split, batch_size, im_size, level, dataset_only=True)

    return DataLoader(
        utilsdata.ConcatDataset([ID_test_dataset + OOD_test_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        timeout=1000
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
        timeout=1000
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
        timeout=1000
    )


def dogs_places_cars(im_size, batch_size):

    datasets = course_ood(im_size, batch_size, datasets_only=True)

    return [DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        timeout=1000
    ) for dataset in datasets]


################################################################
#                      Lightning Loaders                       #
################################################################


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





################################################################
#                       Distributed Data                       #
################################################################





def distributed_training_data(
    split: str, batch_size: int, im_size: str, ood_ratio: int, rank: int, world_size:int
) -> Tuple[DataLoader, DataLoader]:
    
    transform_train = get_transforms(im_size)

    DATASET_ROOT = dataset_root(split)

    ID_train_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_trainset.txt", transform_train, is_fgcvaircraft=True
    )

    OOD_train_dataset = ImageFolder(
        root="/data/imagenet/ILSVRC/Data/CLS-LOC/train/",
        transform=transform_train,
        target_transform=pytorch_ood.utils.ToUnknown(),
    )

    ID_sampler = utilsdata.DistributedSampler(
        ID_train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    OOD_sampler = utilsdata.DistributedSampler(
        OOD_train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    ID_loader = DataLoader(
        ID_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=ID_sampler,
        timeout=1000
    )

    OOD_loader = make_infinite( # Make infinite generator
        DataLoader(
            OOD_train_dataset,
            batch_size=ood_ratio * batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
            sampler=OOD_sampler,
            timeout=1000
        )
    )

    return ID_loader, OOD_loader




def distributed_testing_data(
    split: str, batch_size: int, im_size: str, rank: int, world_size: int, level: int = 1
) -> DataLoader:
    """Get the testing data

    Args:
        split (str):  one of ['split1', 'split2', 'split3', 'split4']
        batch_size (int): Batch Size
        im_size (str): Image Size
        level (int, optional): OOD classification Level, one of [1, 2, 3, -1]. Defaults to 1.
             1 for manufacturer
             2 for variant
             3 for model
            -1 for all.

    Returns:
        DataLoader: Dataloader of loaded data
    """

    transform_test = get_transforms(im_size, False)

    DATASET_ROOT = dataset_root(split)

    ID_test_dataset = txtfiledataset.Dataset_fromTxtFile(
        DATASET_ROOT + "/ID_testset.txt", transform_test, is_fgcvaircraft=True
    )

    OOD_test_dataset = txtfiledataset.Dataset_fromTxtFile_with_levels(
        DATASET_ROOT + "/OOD_testset.txt",
        transform_test,
        target_transform=pytorch_ood.utils.ToUnknown(),
        is_fgcvaircraft=True,
    )
    OOD_test_dataset.set_level(level)

    concat_set = utilsdata.ConcatDataset([ID_test_dataset + OOD_test_dataset]),
    concat_sampler = utilsdata.DistributedSampler(concat_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)


    return DataLoader(
        utilsdata.ConcatDataset([ID_test_dataset + OOD_test_dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        sampler=concat_sampler,
        timeout=1000
        
    )
