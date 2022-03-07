import os
import glob
import torch
import rasterio
import numpy as np
from torch import Tensor
from einops import rearrange
import kornia.augmentation as K
import torchvision.transforms as T
from rasterio.enums import Resampling
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Dict, List, Optional, Sequence


class Geo2022(Dataset):
    classes = [
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    ]
    
    def __init__(
        self,
        files: List[Dict[str, str]],
        label: bool = True,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        image_size: Optional[List[int]] = None,
    ):
        self.files = files
        self.label = label
        self.transforms = transforms
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.image_size = image_size

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        files = self.files[index]
        if self.image_size:
            image = self._load_image(files["image"], shape=self.image_size)
        else:
            image = self._load_image(files["image"])
            
        dem = self._load_image(files["dem"], shape=image.shape[1:])
        image = torch.cat(tensors=[image, dem], dim=0)

        sample = {"image": image}

        if self.label:
            mask = self._load_target(files["target"], shape=image.shape[1:])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Load a single image.
        Args:
            path: path to the image
            shape: the (h, w) to resample the image to
        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor: Tensor = torch.from_numpy(array)
            return tensor

    def _load_target(self, path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Load the target mask for a single image.
        Args:
            path: path to the image
        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_shape=shape, out_dtype="int32", resampling=Resampling.nearest
            )
            tensor: Tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor


class  GEONRW_DataSet:
    dem_min, dem_max = -79.18, 3020.26
    dem_nodata = -99999.0
    metadata = {
        "train": "labeled_train",
        "train-unlabeled": "unlabeled_train",
        "val": "val",
    }
    image_root = "BDORTHO"
    dem_root = "RGEALTI"
    target_root = "UrbanAtlas"

    def __init__(
            self,
            root_dir: str,
            train_batch: int = 8,
            val_batch: int = 4,
            num_workers: int = 0,
            val_split_pct: float = 0.1,
            patch_size: int = 256,
            gpu_num: int = 6,
            shuffle_dataset: bool = True
    ):
        self.root_dir = root_dir
        self.gpu_num = gpu_num
        self.train_batch = int(train_batch / self.gpu_num) * self.gpu_num
        self.val_batch = int(val_batch / self.gpu_num) * self.gpu_num
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.shuffle_dataset = shuffle_dataset

        self.train_label_files, self.val_label_files = self._generate_label_files()
        self.train_unlabel_files = self._generate_unlabel_files()

    def _load_files(self, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.
        Returns:
            list of dicts containing paths for each pair of image/dem/mask
        """
        directory = os.path.join(self.root_dir, self.metadata[split])
        images = glob.glob(
            os.path.join(directory, "**", self.image_root, "*.tif"), recursive=True
        )

        files = []
        for image in sorted(images):
            dem = image.replace(self.image_root, self.dem_root)
            dem = f"{os.path.splitext(dem)[0]}_RGEALTI.tif"

            if split == "train":
                target = image.replace(self.image_root, self.target_root)
                target = f"{os.path.splitext(target)[0]}_UA2012.tif"
                files.append(dict(image=image, dem=dem, target=target))
            else:
                files.append(dict(image=image, dem=dem))

        return files

    def _generate_label_files(self):
        label_files = self._load_files("train")
        if self.shuffle_dataset:
            np.random.shuffle(label_files)

        split = int(len(label_files) * self.val_split_pct / self.gpu_num) * self.gpu_num
        train_label_files, val_label_files = label_files[split:], label_files[:split]
        return train_label_files, val_label_files

    def _generate_unlabel_files(self):
        unlabel_files = self._load_files("train-unlabeled")
        return unlabel_files

    def preprocess(self, sample):
        # RGB is uint8 so divide by 255
        sample['image'][:3] /= 255.0
        sample['image'][-1] = (sample['image'][-1] - self.dem_min) / (
                self.dem_max - self.dem_min
        )
        sample['image'][-1] = torch.clip(sample['image'][-1], min=0.0, max=1.0)

        if 'mask' in sample:
            # ignore the clouds and shadows class (not used in scoring)
            sample['mask'][sample['mask'] == 15] = 0
            sample['mask'] = rearrange(sample['mask'], 'h w -> () h w')

        return sample

    def crop(self, sample):
        if 'mask' in sample:
            random_crop = K.AugmentationSequential(
                K.RandomCrop((self.patch_size * 3, self.patch_size * 2), p=1.0, keepdim=False),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=['input', 'mask'],
            )
            sample['mask'] = sample['mask'].to(torch.float)
            sample['image'], sample['mask'] = random_crop(sample['image'], sample['mask'])
            sample['mask'] = sample['mask'].to(torch.long)
            sample['image'] = rearrange(sample['image'], '() c h w -> c h w')
            sample['mask'] = rearrange(sample['mask'], '() c h w -> c h w')
        else:
            random_crop = K.AugmentationSequential(
                K.RandomCrop((self.patch_size * 3, self.patch_size * 2), p=1.0, keepdim=False),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=['input'],
            )
            sample['image'] = random_crop(sample['image'])
            sample['image'] = rearrange(sample['image'], '() c h w -> c h w')
        return sample

    def train_label_dataloader(self):
        transforms = T.Compose([self.preprocess, self.crop])
        train_label_dataset = Geo2022(self.train_label_files, label=True, transforms=transforms)
        print('train_label_dataset', len(train_label_dataset))
        return DataLoader(train_label_dataset, batch_size=self.train_batch, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def train_unlabel_dataloader(self):
        unlabel_percent = len(self.train_unlabel_files) / (len(self.train_label_files) + len(self.train_unlabel_files))
        bs = self.train_batch * unlabel_percent / (self.gpu_num * (1. - unlabel_percent))
        bs = min(max(round(bs) * self.gpu_num, self.gpu_num), self.train_batch * self.gpu_num)
        transforms = T.Compose([self.preprocess, self.crop])
        train_unlabel_dataset = Geo2022(self.train_unlabel_files, label=False, transforms=transforms)
        print('train_unlabel_dataset', len(train_unlabel_dataset))
        return DataLoader(train_unlabel_dataset, batch_size=bs, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_label_dataloader(self):
        transforms = T.Compose([self.preprocess])
        val_label_dataset = Geo2022(self.val_label_files, label=True, transforms=transforms, image_size=(2000, 2000))
        print('val_label_dataset', len(val_label_dataset))
        return DataLoader(val_label_dataset, batch_size=self.val_batch, num_workers=self.num_workers,
                          shuffle=False, drop_last=False)
