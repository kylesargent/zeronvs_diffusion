# https://github.com/webdataset/webdataset/tree/e4c30ef816de7567fe537467938c82005702d613

import numpy as np
import imageio
import io
from tqdm import tqdm
import cv2
import webdataset as wds
from ldm.data import common
import functools
import json
import math
import os
from google.cloud import storage
import collections
import gc
from torch.utils.data import IterableDataset
from webdataset import filters

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from PIL import Image
import matplotlib.pyplot as plt
import torch
from urllib.parse import urlparse
import pickle
import warnings
import numpy as np
import scipy
from ldm.data import webdataset_base
from ldm.data import webdataset_co3d
from ldm.data import webdataset_re10k


warnings.filterwarnings(action="ignore", category=UserWarning, module="google")
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="scipy")


# OBJAVERSE_PUBLIC_URL = "https://storage.googleapis.com/viscam-cloud-storage-central1/kylesargent/views_release_raw.tar"
# DATASET_ZOO = {"objaverse": {"compose_fn": objaverse_compose_fn}}


def rtmv_compose_fn(samples, rate):
    # rate*l pairs will be sampled from a scene of length l
    # lower rates have more unbiased sampling but fewer imgs/s
    cur_uid = None
    uid_ctr = 0
    scene = get_new_scene()

    for sample in samples:
        _, uid, str_idx = sample["__key__"].split("/")
        if str_idx in {"nerf_train", "nerf_val", "nerf_test", "scene"}:
            continue

        if cur_uid != uid:
            uid_ctr += 1

            # print(cur_uid)
            # print(int(cur_uid or -1) in subset_info)
            # print(subset_info.keys())
            # if cur_uid is not None and int(cur_uid) in subset_info:
            if cur_uid is not None:
                print("yield from scene ", cur_uid)
                yield from get_pairs(scene, rate, cur_uid, unique_source_view=True)

            scene = get_new_scene()
            cur_uid = uid

        idx = int(str_idx)

        scene["images"][idx] = sample["exr"]

        camera_data = json.loads(sample["json"])["camera_data"]
        # print(camera_data.keys())
        world2cam = np.array(camera_data["camera_view_matrix"]).T[:3]
        assert world2cam.shape == (3, 4)

        scene["cams"][idx] = world2cam

    yield from get_pairs(scene, rate, cur_uid, unique_source_view=True)


def get_image_from_bytes_rtmv(image_bytes):
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    assert h == w

    # TODO(kylesargent): check if color channels are right here
    img = cv2.resize(img[..., :3], (256, 256))
    img = img * 2 - 1
    return img


def split_by_node(urls):
    node_id, node_count = (
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
    )
    node_urls = urls[node_id::node_count]
    print(f"node id {node_id} got {len(node_urls)} shards:")
    print(node_urls)
    return node_urls


def split_by_worker(urls):
    node_id, node_count = (
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
    )
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        print(f"inside worker {wi.id} rank {node.id}")
        return urls[wi.id :: wi.num_workers]


WORKER_TO_FDS = collections.defaultdict(list)


def _patch_gopen(url, mode="rb", bufsize=8192, **kw):
    gc.collect()

    # url = wds.gopen.rewrite_url(url)
    pr = wds.gopen.urlparse(url)
    assert pr.scheme == "gs"
    client = storage.Client()

    worker_info = torch.utils.data.get_worker_info()

    # print("initializing gcs client at ", (url, worker_info))
    chunk_size = 104857600  # 100MB
    # chunk_size = chunk_size * 5

    # def get_blob():
    #     return storage.Blob(
    #         pr.path[1:], client.get_bucket(pr.netloc)
    #     ).open(mode)

    # return get_blob()
    # worker_info.id
    if worker_info is not None:
        worker_id = worker_info.id
    else:
        worker_id = 0

    # while WORKER_TO_FDS[worker_id]:
    #     fd = WORKER_TO_FDS[worker_id].pop()
    #     # print("closing old file handle", fd)
    #     fd.close()

    fd = storage.Blob(pr.path[1:], client.get_bucket(pr.netloc)).open(mode)
    # WORKER_TO_FDS[worker_id].append(fd)
    return fd

    # state = {}
    # state[worker_info.id] = get_blob()
    # return state[worker_info.id]


def _compute_n_batches(config):
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1

    n_pairs = int(
        config["dataset_n_scenes"]
        / world_size
        * config["views_per_scene"]
        * config["rate"]
        * max(config["num_workers"], 1)
    )
    n_batches = int(n_pairs // config["batch_size"])
    return n_batches


def get_dataset(
    dataset_url,
    dataset_n_shards,
    rate,
    dataset_name,
    shuffle_buffer_size,
    resampled=True,
    compute_nearplane_quantile=False,
    yield_scenes=False,
    **kwargs,
):
    if dataset_url.startswith("gs://"):
        # raise
        print("patching gopen")
        wds.gopen.gopen = _patch_gopen

    all_dataset_urls = [
        dataset_url.format(shard=shard) for shard in range(dataset_n_shards)
    ]
    print("all urls", all_dataset_urls)

    print("Made a loader!")

    try:
        print(torch.distributed.get_rank(), torch.distributed.get_world_size())
        dataset_urls = split_by_node(all_dataset_urls)
        assert len(dataset_urls) > 0
    except RuntimeError:
        print("Must initialize DDP for node sharding.")
        dataset_urls = all_dataset_urls
    except AssertionError:
        print("Node got 0 shards, falling back to full dataset.")
        dataset_urls = all_dataset_urls

    # https://github.com/webdataset/webdataset/blob/e4c30ef816de7567fe537467938c82005702d613/webdataset/compat.py#L86
    ds = wds.WebDataset(
        dataset_urls,
        resampled=resampled,
        shardshuffle=resampled,
        handler=wds.warn_and_continue,
        # nodesplitter=my_split_by_node,
    )

    # import pdb
    # pdb.set_trace()

    if dataset_name == "objaverse":
        raise NotImplementedError
    elif dataset_name == "rtmv":
        raise NotImplementedError
    elif dataset_name == "co3d":
        ds_interface = webdataset_co3d.CO3D(
            rate=rate, compute_nearplane_quantile=compute_nearplane_quantile
        )
    elif dataset_name == "re10k":
        ds_interface = webdataset_re10k.RE10K(
            rate=rate, compute_nearplane_quantile=compute_nearplane_quantile
        )
    elif dataset_name == "acid":
        # same interface as re10k
        ds_interface = webdataset_re10k.RE10K(
            rate=rate, compute_nearplane_quantile=compute_nearplane_quantile
        )
    else:
        raise NotImplementedError

    ds = ds.compose(functools.partial(ds_interface.compose_fn, yield_scenes=yield_scenes))

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    # need to batch in the dataset to avoid too many open file handles
    if yield_scenes:
        ds = ds.map(ds_interface.get_combined_scene_data)
    else:
        ds = ds.map(ds_interface.preprocess_tuple)
    return ds


# class SampleEqually(IterableDataset):
#     def __init__(self, datasets):
#         super().__init__()
#         self.datasets = datasets

#     def __iter__(self):
#         sources = [iter(ds) for ds in self.datasets]
#         while True:
#             for source in sources:
#                 try:
#                     yield next(source)
#                 except StopIteration:
#                     return


# print("debugging worker info")
class SampleEquallyBatched(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(self, datasets, probabilities, batch_size):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.probabilities = probabilities

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        worker_info = torch.utils.data.get_worker_info()

        while True:
            try:
                dataset_idxs = np.random.choice(
                    a=list(range(len(self.datasets))),
                    size=self.batch_size,
                    p=self.probabilities,
                )
                items = []
                for dataset_idx in dataset_idxs:
                    item = next(sources[dataset_idx])

                    # item = webdataset_base.batch_struct_from_tuple(item)
                    # item['uid'] = worker_info.id
                    # item = webdataset_base.batch_struct_to_tuple(item)

                    items.append(item)

                batch = filters.default_collation_fn(items)
                yield batch
            except StopIteration:
                return


def get_loader(
    num_workers,
    batch_size,
    shuffle_buffer_size,
    resampled=True,
    prefetch_factor=4,
    **kwargs,
):
    dataset_configs = []
    for kwarg in kwargs:
        if kwarg.startswith("dataset_config_"):
            dataset_config = kwargs[kwarg]
            dataset_config["batch_size"] = batch_size
            dataset_config["num_workers"] = num_workers
            dataset_configs.append(dataset_config)

    # import pdb
    # pdb.set_trace()
    datasets = [
        get_dataset(**dataset_config, shuffle_buffer_size=shuffle_buffer_size)
        for dataset_config in dataset_configs
    ]
    n_batches_all = [
        _compute_n_batches(dataset_config) for dataset_config in dataset_configs
    ]
    n_batches = sum(n_batches_all)
    print("n_batches_all=", n_batches_all)
    print("n_batches=", n_batches)
    probabilities = [dataset_config.probability for dataset_config in dataset_configs]

    # import pdb
    # pdb.set_trace()

    ds = SampleEquallyBatched(
        datasets=datasets, probabilities=probabilities, batch_size=batch_size
    )
    # datasets = [ds.batched(batch_size) for ds in datasets]
    # ds = SampleEqually(datasets)  # .batched(batch_size)

    # def collation_fn(*args, **kwargs):
    #     import pdb
    #     pdb.set_trace()

    #     1 + 1

    # raise
    # ds = ds.batched(batch_size, collation_fn=collation_fn)
    # ds = datasets[0].batched(batch_size)

    # import pdb
    # pdb.set_trace()

    # TODO(kylesargent): add mixing logic here
    # ds = ds.batched(batch_size)
    # ds = ds.compose(
    #     filters.batched(
    #         batchsize, collation_fn=filters.default_collation_fn, partial=True
    #     )
    # )

    if num_workers > 0:
        prefetch_factor = prefetch_factor
    else:
        # webdataset has a very bad system for this; namely, if it detects that the # of workers is >0, the prefetch
        # factor needs to not be specified, i.e. be equal to the default value of 2...
        prefetch_factor = 2

    loader = wds.WebLoader(
        ds, num_workers=num_workers, batch_size=None, prefetch_factor=prefetch_factor
    )

    # loader = loader.unbatched().shuffle(1000).batched(batch_size)

    loader = loader.map(webdataset_base.batch_struct_from_tuple)
    loader = loader.with_epoch(n_batches)

    return loader
