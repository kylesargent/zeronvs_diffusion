import numpy as np
from PIL import Image
from ldm.data import webdataset_base
from ldm.data import common
import json
import io
import scipy
import imageio
import pyspng


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0],
        ]
    )


def _load_16big_png_depth(depth_pil):
    # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
    # we cast it to uint16, then reinterpret as float16, then cast to float32
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    return depth


def get_fov_degs(*, metadata):
    assert tuple(metadata["principal_point"]) == (0.0, 0.0)
    # image boundaries are [-1, 1] x [-1, 1]

    focal_y, focal_x = metadata["focal_length"]
    field_of_view_deg_x = 2 * np.arctan2(1, focal_x) * (180 / np.pi)
    field_of_view_deg_y = 2 * np.arctan2(1, focal_y) * (180 / np.pi)
    return field_of_view_deg_x, field_of_view_deg_y


def get_pixtocam(focal: float, width: float, height: float):
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * 0.5, height * 0.5)
    return np.linalg.inv(camtopix)


def fov_to_intrinsics_matrix(fov_deg, height):
    # assuming a square camera and fov in degrees and image with "height" pixels,
    # get the intrinsics matrix in our canonical format
    fov = np.deg2rad(fov_deg)
    focal = height / (2 * np.tan(fov / 2))
    pixtocam = get_pixtocam(focal, height, height)

    # homogenize it
    eye = np.eye(4)
    eye[:3, :3] = pixtocam
    return eye


def _get_camera_matrix(viewpoint):
    R = np.array(viewpoint.R)
    T = np.array(viewpoint.T)[:, None]
    RT = np.concatenate([R.T, T], axis=1)
    assert RT.shape == (3, 4)

    # pytorch3d -> opencv
    RT[0] *= -1
    RT[1] *= -1

    # OpenCV -> OpenGL
    RT[1] *= -1
    RT[2] *= -1

    return RT


def central_crop_img_arr(img):
    h, w, c = img.shape
    assert min(h, w) == 256
    s = 256
    oh_resid = (h - s) % 2
    ow_resid = (w - s) % 2
    oh = (h - s) // 2
    ow = (w - s) // 2
    img = img[oh : h - oh - oh_resid, ow : w - ow - ow_resid]
    assert img.shape == (256, 256, c), img.shape
    return img


def get_image_from_bytes_co3d(image_bytes):
    # with io.BytesIO(image_bytes) as fp:
    #     img = imageio.imread(fp)
    img = pyspng.load(image_bytes)
    img = img / 255.0

    # import pdb
    # pdb.set_trace()

    # central crop
    img = central_crop_img_arr(img)

    img = img * 2 - 1
    img = np.array(img)
    return img


def get_depth_from_bytes_co3d(depth_bytes):
    with io.BytesIO(depth_bytes) as fp:
        depth_pil = Image.open(fp)
        depth_arr = _load_16big_png_depth(depth_pil)
    depth_arr = central_crop_img_arr(depth_arr[..., None])
    depth_arr = depth_arr.astype(np.float32)

    # clip the occasional invalid negative depth
    depth_arr = np.clip(depth_arr, 0, None)

    return depth_arr


def central_crop_v2(image):
    h, w = image.size
    s = min(h, w)
    # print(s)
    oh = (h - s) // 2
    oh_resid = (h - s) % 2
    ow = (w - s) // 2
    ow_resid = (w - s) % 2
    crop_bounds = [oh, ow, h - oh - oh_resid, w - ow - ow_resid]
    # print(crop_bounds)
    new_image = image.crop(crop_bounds)
    assert new_image.size == (s, s), (image.size, (s, s), new_image.size)
    return new_image


def fill_zeros_nearest_nonzeros(array):
    zero_mask = array == 0

    indices = scipy.ndimage.distance_transform_edt(
        zero_mask, return_distances=False, return_indices=True
    )
    filled_array = array[tuple(indices)]
    return filled_array


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def get_world_data(*, worldtocams):
    cams = worldtocams
    assert cams.ndim == 3
    assert cams.shape[1:] == (3, 4)
    h = np.array([0, 0, 0, 1])[None, None]
    h = np.broadcast_to(h, (cams.shape[0], h.shape[1], h.shape[2]))
    cams = np.concatenate([cams, h], axis=1)
    camtoworld = np.linalg.inv(cams)

    focus_pt = focus_point_fn(camtoworld)[None]

    locations = camtoworld[:, :3, -1]
    assert locations.shape == (cams.shape[0], 3)
    center = locations.mean(axis=0, keepdims=True)
    assert focus_pt.shape == center.shape, (focus_pt.shape, center.shape)
    scene_radius = np.linalg.norm(locations - center, axis=1).mean()
    scene_radius_focus_pt = np.linalg.norm(locations - focus_pt, axis=1).mean()
    cams = cams[:, :3]

    camera_loc = camtoworld[:, :3, -1]
    center_of_mass = camera_loc.mean(axis=0, keepdims=True)

    focus_pt_to_center_of_mass = np.linalg.norm(focus_pt - center_of_mass)
    camera_to_center_of_mass = np.linalg.norm((camera_loc - center_of_mass), axis=-1)
    elevation = np.arctan2(focus_pt_to_center_of_mass, camera_to_center_of_mass)
    elevation_deg = elevation * 180 / np.pi

    mean_elevation_deg = elevation_deg.mean()

    return {
        "center": center,
        "focus_pt": focus_pt,
        "scene_radius_focus_pt": scene_radius_focus_pt,
        "scene_radius": scene_radius,
        "camtoworld": camtoworld,
        "elevation_deg": elevation_deg,
        "mean_elevation_deg": mean_elevation_deg,
        "scale_adjustment": 1.0,
    }


class CO3D(webdataset_base.WebDatasetBase):
    def compose_fn(self, samples, yield_scenes=False):
        # rate*l pairs will be sampled from a scene of length l
        # lower rates have more unbiased sampling but fewer imgs/s
        rate = self.rate

        cur_uid = None
        uid_ctr = 0
        scene = webdataset_base.get_new_scene()

        for sample in samples:
            # print(uid_ctr)
            *_, category_uid, scene_uid, str_idx = sample["__key__"].split("/")
            uid = f"{category_uid}_{scene_uid}"
            # print(sample.keys())

            if str_idx.endswith("_depth"):
                is_depth = True
                str_idx = str_idx[: -len("_depth")]
            else:
                is_depth = False

            if cur_uid != uid:
                uid_ctr += 1

                if uid_ctr % 10_000 == 0:
                    print(f"Processing scene: {uid_ctr}!")

                if cur_uid is not None:
                    if yield_scenes:
                        scene['uid'] = cur_uid
                        yield scene
                    else:
                        yield from self.get_tuples(scene, cur_uid)

                scene = webdataset_base.get_new_scene()
                cur_uid = uid

            assert str_idx.startswith("frame")
            str_idx = str_idx[len("frame") :]
            if str_idx.endswith("_metadata"):
                str_idx = str_idx[: -len("_metadata")]

            idx = int(str_idx)
            idx -= 1  # co3d view idxs start at 1, need to adjust it

            if "png" in sample and not is_depth:
                scene["images"][idx] = sample["png"]
            if "png" in sample and is_depth:
                scene["depths"][idx] = sample["png"]

            if "json" in sample:
                camera, metadata = self.load_sample_metadata(sample["json"])
                scene["cams"][idx] = camera
                scene["metadatas"][idx] = metadata

            if "json" not in sample and "png" not in sample:
                raise NotImplementedError

    def load_sample_metadata(self, metadata_bytes):
        metadata = json.load(io.BytesIO(metadata_bytes))
        camera = np.concatenate(
            [np.array(metadata["R"]).T, np.array(metadata["T"])[:, None]],
            axis=1,
        )
        return camera, metadata

    def get_combined_scene_data(self, scene):
        # sorted valid view idxs
        true_idx_to_idx = np.array(sorted(scene["images"]))
        l = len(scene["images"])
        if l == 0:
            return
        assert len(true_idx_to_idx) == l

        try:
            assert set(scene["images"]) == set(scene["cams"]), (
                set(scene["images"]),
                set(scene["cams"]),
            )
        except AssertionError as e:
            print("corruped scene!")
            print(set(scene["images"]))
            print(set(scene["cams"]))
            return

        cams = np.array([scene["cams"][true_idx_to_idx[i]] for i in range(l)])

        # up=y,right=x,forward=z
        # pytorch3d format -> OpenCV
        cams[:, 0] *= -1
        cams[:, 1] *= -1

        # OpenCV -> OpenGL
        cams[:, 1] *= -1
        cams[:, 2] *= -1

        co3d_metadata = next(iter(scene["metadatas"].values()))

        world_data = get_world_data(worldtocams=cams)
        world_data["scale_adjustment"] = co3d_metadata["scale_adjustment"]

        fov_deg = min(get_fov_degs(metadata=co3d_metadata))

        camera_data = {"fov_deg": fov_deg}

        metadata = {"true_idx_to_idx": true_idx_to_idx}
        return camera_data, world_data, metadata, scene

    def get_tuples(self, scene, uid):
        all_scene_data = self.get_combined_scene_data(scene)
        if all_scene_data is None:
            return None
        else:
            camera_data, world_data, metadata, _ = all_scene_data

        l = len(scene["images"])
        rate = self.rate
        if isinstance(rate, float):
            n_pairs = int(rate * l)
        elif isinstance(rate, str):
            assert rate.endswith("_per_scene")
            n_pairs = int(rate[0])
        else:
            raise NotImplementedError

        pair_idxs = [
            np.random.choice(np.arange(l), (2,), replace=False) for _ in range(n_pairs)
        ]
        true_idx_to_idx = metadata["true_idx_to_idx"]

        if self.compute_nearplane_quantile:
            depths = [
                get_depth_from_bytes_co3d(depth) * world_data["scale_adjustment"]
                for depth in scene["depths"].values()
            ]
            masked_depths = [depth[depth != 0] for depth in depths]
            quantiles = [
                np.quantile(depth, 0.05) if len(depth) > 0 else None
                for depth in masked_depths
            ]
            
            if all(quantile is None for quantile in quantiles):
                print("got scene with no points! this should not happen often!")
                return
            else:
                nonempty_quantiles = np.array(
                    [quantile for quantile in quantiles if quantile is not None]
                )
                nearplane_quantile = np.quantile(nonempty_quantiles, 0.1)
        else:
            nearplane_quantile = None

        # import pdb
        # pdb.set_trace()

        for i, j in pair_idxs:
            pair_uid = "%.4d__%.4d__%s" % (i, j, uid)

            batch_struct = webdataset_base.get_batch_struct(
                image_target=scene["images"][true_idx_to_idx[i]],
                image_cond=scene["images"][true_idx_to_idx[j]],
                depth_target=scene["depths"][true_idx_to_idx[i]],
                depth_target_filled=None,
                depth_cond=scene["depths"][true_idx_to_idx[j]],
                depth_cond_filled=None,
                uid=uid,
                pair_uid=pair_uid,
                T=None,
                target_cam2world=world_data["camtoworld"][i],
                cond_cam2world=world_data["camtoworld"][j],
                center=world_data["center"],
                focus_pt=world_data["focus_pt"],
                scene_radius=world_data["scene_radius"],
                scene_radius_focus_pt=world_data["scene_radius_focus_pt"],
                fov_deg=camera_data["fov_deg"],
                scale_adjustment=world_data["scale_adjustment"],
                nearplane_quantile=nearplane_quantile,
                depth_cond_quantile25=None,
                cond_elevation_deg=world_data['elevation_deg'][j]
            )
            yield webdataset_base.batch_struct_to_tuple(batch_struct)

    def preprocess_tuple(self, tuple):
        batch_struct = webdataset_base.batch_struct_from_tuple(tuple)

        # import pdb
        # pdb.set_trace()

        batch_struct["depth_target"] = get_depth_from_bytes_co3d(
            batch_struct["depth_target"]
        )
        batch_struct["depth_target"] *= batch_struct["scale_adjustment"]
        batch_struct["depth_target_filled"] = None

        batch_struct["depth_cond"] = get_depth_from_bytes_co3d(
            batch_struct["depth_cond"]
        )
        batch_struct["depth_cond_filled"] = None
        
        batch_struct["depth_cond"] *= batch_struct["scale_adjustment"]

        # batch_struct["depth_cond_quantile25"] = np.quantile(
        #     batch_struct["depth_cond"], 0.25
        # )

        batch_struct["image_target"] = get_image_from_bytes_co3d(
            batch_struct["image_target"]
        ).astype(np.float32)
        batch_struct["image_cond"] = get_image_from_bytes_co3d(
            batch_struct["image_cond"]
        ).astype(np.float32)

        batch_struct["T"] = T = common.get_T(
            np.linalg.inv(batch_struct["target_cam2world"])[:3],
            np.linalg.inv(batch_struct["cond_cam2world"])[:3],
            to_torch=False,
        ).astype(np.float32)

        # T = common.get_T(world2cam_i, world2cam_j, to_torch=False).astype(np.float32)
        # data["T_full"] = np.stack([world2cam_i, world2cam_j])
        # target_cam2world = np.linalg.inv(
        #     np.concatenate([world2cam_i, np.array([0, 0, 0, 1.0])[None]], axis=0)
        # )
        # cond_cam2world = np.linalg.inv(
        #     np.concatenate([world2cam_j, np.array([0, 0, 0, 1.0])[None]], axis=0)
        # )

        return webdataset_base.batch_struct_to_tuple(batch_struct)

    def tuple_to_dict(self, tuple):
        return webdataset_base.batch_struct_from_tuple(tuple)
