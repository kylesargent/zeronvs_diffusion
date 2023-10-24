from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.data import webdataset_utils
from ldm.data import common
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import pickle


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)


def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [
        NfpDataset(
            x,
            image_transforms=copy.copy(tforms),
            default_caption="A view from a train window",
        )
        for x in dirs
    ]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2 * self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i + 1) * self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[..., ::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[..., ::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption,
        }
        return data


# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ]
    )
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path] * repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [
            FolderData(p, caption_file=c, **kwargs)
            for (p, c) in zip(paths, caption_files)
        ]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)


class NfpDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
    ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1

    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index + 1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size,
        total_view,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        **kwargs,
    ):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if "image_transforms" in dataset_config:
            image_transforms = [
                torchvision.transforms.Resize(dataset_config.image_transforms.size)
            ]
        else:
            image_transforms = []
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return wds.WebLoader(
            ObjaverseData(
                root_dir=self.root_dir,
                total_view=self.total_view,
                validation=self.validation,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def _load_dataset_from_pkl(dataset_url_pkl, batch_size, subsample, scene_scale):
    with open(dataset_url_pkl, "rb") as fp:
        np_batches = pickle.load(fp)

    def unstack(batch):
        return [
            {k: v[i] for (k, v) in batch.items()}
            for i in range(batch["image_cond"].shape[0])
        ]

    np_batches = [datum for batch in np_batches for datum in unstack(batch)]
    for np_batch in np_batches:
        # print(scene_scale)
        np_batch['cond_cam2world'][:3, -1] *= scene_scale
        np_batch['target_cam2world'][:3, -1] *= scene_scale

    # import pdb
    # pdb.set_trace()

    random.shuffle(np_batches)

    if subsample != 1.0:
        np_batches = np_batches[: int(len(np_batches) * subsample)]

    dataloader = torch.utils.data.DataLoader(np_batches, batch_size=batch_size)
    return dataloader


class WDSGenericDataModule(pl.LightningDataModule):
    def __init__(self, train_config, val_config, test_config=None, **kwargs):
        super().__init__(self)
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config

        # compute n_batches
        # for name, config in zip(
        #     ["train", "val", "test"],
        #     [self.train_config, self.val_config, self.test_config],
        # ):
        #     if config is not None:
        #         n_batches = _compute_n_batches(config)
        #         print(f"n_batches={n_batches} for {name}")
        #         config["n_batches"] = n_batches

    def train_dataloader(self):
        return webdataset_utils.get_loader(**self.train_config)

    def val_dataloader(self):
        if "dataset_url_pkl" in self.val_config:
            return _load_dataset_from_pkl(
                self.val_config.dataset_url_pkl,
                batch_size=self.val_config.batch_size,
                subsample=self.val_config.subsample,
                scene_scale=self.val_config.scene_scale
            )
            
        else:
            return webdataset_utils.get_loader(**self.val_config)

    def test_dataloader(self):
        return self.val_dataloader()


class WDSObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        dataset_url,
        dataset_url_valid,
        dataset_url_valid_pkl,
        dataset_n_shards,
        valid_paths_json_url,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        rate=1.0,
        val_rate=1.0,
        shuffle_buffer_size=None,
        subsample_frac=None,
        debug_validate_on_train=False,
        **kwargs,
    ):
        super().__init__(self)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.dataset_url = dataset_url
        self.dataset_url_valid = dataset_url_valid
        self.dataset_url_valid_pkl = dataset_url_valid_pkl
        self.dataset_n_shards = dataset_n_shards
        self.valid_paths_json_url = valid_paths_json_url
        self.shuffle_buffer_size = shuffle_buffer_size
        self.subsample_frac = subsample_frac
        self.rate = rate
        self.val_rate = val_rate
        self.debug_validate_on_train = debug_validate_on_train

        if shuffle_buffer_size * num_workers > 10_000:
            buf_bytes_approx = (
                shuffle_buffer_size * num_workers * 256**2 * 3 * 4 * 2 / 1e9
            )
            print(f"warning: shuffle buffer is approximately {buf_bytes_approx} GB")

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if "image_transforms" in dataset_config:
            image_transforms = [
                torchvision.transforms.Resize(dataset_config.image_transforms.size)
            ]
        else:
            image_transforms = []
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

        with open(valid_paths_json_url) as fp:
            scene_uids = json.load(fp)

        n_scenes = len(scene_uids)
        n_pairs = int(n_scenes * 12 * rate * num_workers)
        n_train_batches = int(n_pairs // batch_size)

        # print("hardcoded n_train_batches!!!!!")
        # n_train_batches = 10

        n_val_batches = 20 * 16 // self.val_batch_size  # hardcoded

        train_summary = dict(
            n_scenes=n_scenes, n_pairs=n_pairs, n_train_batches=n_train_batches
        )
        print("train summary", train_summary)
        val_summary = dict(n_val_batches=n_val_batches)
        print("val_summary", val_summary)

        self.train_loader_kwargs = dict(
            dataset_url=self.dataset_url,
            dataset_n_shards=self.dataset_n_shards,
            valid_paths_json_url=self.valid_paths_json_url,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            transforms=self.image_transforms,
            split="train",
            shuffle_buffer_size=self.shuffle_buffer_size,
            subsample_frac=self.subsample_frac,
            rate=self.rate,
            n_batches=n_train_batches,
        )

        self.val_loader_kwargs = dict(
            dataset_url=self.dataset_url_valid,
            dataset_n_shards=self.dataset_n_shards,
            valid_paths_json_url=self.valid_paths_json_url,
            num_workers=0,  # TODO(kylesargent): fix this when we have an actual validation sharding scheme
            batch_size=self.val_batch_size,
            transforms=self.image_transforms,
            split="validation",
            shuffle_buffer_size=self.shuffle_buffer_size,
            rate=self.val_rate,
            dataset_name="rtmv",
            n_batches=n_val_batches,
            resampled=False,
        )

    def train_dataloader(self):
        return webdataset_utils.get_loader(**self.train_loader_kwargs)

    def val_dataloader(self):
        if self.debug_validate_on_train:
            val_loader_kwargs = dict(**self.train_loader_kwargs)
            val_loader_kwargs["n_batches"] = 10
            return webdataset_utils.get_loader(**val_loader_kwargs)

        if self.dataset_url_valid_pkl is None:
            return webdataset_utils.get_loader(**self.val_loader_kwargs)
        else:
            return _load_dataset_from_pkl(self.dataset_url_valid_pkl)

    def test_dataloader(self):
        return self.val_dataloader()


class ObjaverseData(Dataset):
    def __init__(
        self,
        root_dir=".objaverse/hf-objaverse-v1/views",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, "valid_paths.json")) as f:
            self.paths = json.load(f)

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 99.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 99.0)
            ]  # used first 99% as training
        print("============= length of dataset %d =============" % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        return common.cartesian_to_spherical(xyz)

    def get_T(self, target_RT, cond_RT):
        return common.get_T(target_RT, cond_RT)

    def load_im(self, path, color):
        """
        replace background pixel with random color in rendering
        """
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.0] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = 12
        index_target, index_cond = random.sample(
            range(total_view), 2
        )  # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)

        color = [1.0, 1.0, 1.0, 1.0]

        try:
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_cond), color)
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(
                self.root_dir, "692db5f2d3a04bb286cb977a7dba903e_1"
            )  # this one we know is valid
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_cond), color)
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class FolderData(Dataset):
    def __init__(
        self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir / chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


import random


class TransformDataset:
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }

    def __getitem__(self, index):
        data = self.ds[index]

        im = data["image"]
        im = im.permute(2, 0, 1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1, 2, 0)

        data["image"] = im
        data["txt"] = data["txt"] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)


def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split="train",
    image_key="image",
    caption_key="txt",
):
    """Make huggingface dataset with appropriate list of transforms applied"""
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert (
        image_column in ds.column_names
    ), f"Didn't find column {image_column} in {ds.column_names}"
    assert (
        text_column in ds.column_names
    ), f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds


class TextOnly(Dataset):
    def __init__(
        self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1
    ):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus * [x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2.0 - 1.0, "c h w -> h w c")
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, "rt") as f:
            captions = f.readlines()
        return [x.strip("\n") for x in captions]


import random
import json


class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir / retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
