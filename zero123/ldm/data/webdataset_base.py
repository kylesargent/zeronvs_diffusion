from typing import Any, Dict
import inspect


def get_new_scene():
    return {"images": {}, "cams": {}, "metadatas": {}, "depths": {}}


BATCH_STRUCT_KEYS = [
    "image_target",
    "image_cond",
    "depth_target",
    "depth_target_filled",
    "depth_cond",
    "depth_cond_filled",
    "uid",
    "pair_uid",
    "T",
    "target_cam2world",
    "cond_cam2world",
    "center",
    "focus_pt",
    "scene_radius_focus_pt",
    "scene_radius",
    "fov_deg",
    "scale_adjustment",
    "nearplane_quantile",
    "depth_cond_quantile25",
    "cond_elevation_deg",
]


def get_batch_struct(**kwargs):
    assert set(kwargs.keys()) == set(BATCH_STRUCT_KEYS)
    return kwargs


def batch_struct_to_tuple(bs_struct):
    assert set(bs_struct.keys()) == set(BATCH_STRUCT_KEYS)
    return tuple(bs_struct[key] for key in sorted(bs_struct))


def batch_struct_from_tuple(bs_tuple):
    return {k: v for (k, v) in zip(sorted(BATCH_STRUCT_KEYS), bs_tuple)}


class WebDatasetBase(object):
    def __init__(self, rate, compute_nearplane_quantile):
        self.rate = rate
        self.compute_nearplane_quantile = compute_nearplane_quantile

    def get_tuples(self, *, scene, uid):
        # How to convert a scene into an iterator of tuples
        raise NotImplementedError

    def compose_fn(self, *, samples):
        # How to iterate over a tarfile and generate samples in the form of tuples
        raise NotImplementedError

    def preprocess_tuple(self, *, tuple):
        # How to convert a tuple into a valid datapoint. Put operations that may blow up
        # the bytes (e.g. png bytes -> RGB image) here, because this will happen after
        # the dataset shuffle.
        raise NotImplementedError

    def tuple_to_dict(self, *, tuple) -> Dict[str, Any]:
        # Convert the tuple to a Dict after Pytorch collates everything
        raise NotImplementedError
