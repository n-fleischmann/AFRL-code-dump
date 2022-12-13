# NAI

# Custom PyTorch dataset that inputs a text file that is formatted in 2 columns:
# Example line from txt file: /pth/to/img.jpg class_num
# Then reads in the lines and contructs a python list
#  input list = [ ["/pth/to/file.png", class#] , ... ]

import torch
from PIL import Image


class Dataset_fromTxtFile(torch.utils.data.Dataset):
    def __init__(
        self, txtfile, transform=None, target_transform=None, is_fgcvaircraft=False
    ):
        # Read in txtfile and construct dset list
        dset_list = []
        infile = open(txtfile, "r")
        for line in infile:
            tmppath, tmplabel = line.rstrip().split()
            dset_list.append([tmppath, int(tmplabel)])
        infile.close()
        # Set class attributes
        self.dset_list = dset_list
        self.transform = transform
        self.target_transform = target_transform
        self.is_fgcvaircraft = is_fgcvaircraft

    def __len__(self):
        return len(self.dset_list)

    def __getitem__(self, idx):
        path, target = self.dset_list[idx]
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.is_fgcvaircraft:
            # Crop 20px bottom banner on every image
            tmpw, tmph = img.size
            img = img.crop((0, 0, tmpw, tmph - 20))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Dataset_fromTxtFile_with_levels(torch.utils.data.Dataset):
    def __init__(
        self, txtfile, transform=None, target_transform=None, is_fgcvaircraft=False
    ):
        # Read in txtfile and construct dset list

        dset_dict = {1: [], 2: [], 3: []}
        infile = open(txtfile, "r")
        for line in infile:
            tmppath, tmplabel = line.rstrip().split()
            dset_dict[int(tmplabel)].append([tmppath, int(tmplabel)])
        infile.close()

        dset_dict[-1] = dset_dict[1] + dset_dict[2] + dset_dict[3]
        # Set class attributes
        self.dset_dict = dset_dict
        self.transform = transform
        self.target_transform = target_transform
        self.is_fgcvaircraft = is_fgcvaircraft

        self.level = 1

    def set_level(self, level):
        assert level in [1, 2, 3, -1], f"level must be in [1, 2, 3, -1] but got {level}"
        self.level = level

    def __len__(self):
        return len(self.dset_dict[self.level])

    def __getitem__(self, idx):
        path, target = self.dset_dict[self.level][idx]
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.is_fgcvaircraft:
            # Crop 20px bottom banner on every image
            tmpw, tmph = img.size
            img = img.crop((0, 0, tmpw, tmph - 20))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
