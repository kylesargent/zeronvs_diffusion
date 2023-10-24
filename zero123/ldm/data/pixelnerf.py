import os
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import cv2
from ldm.data import common
import torch


import skimage.measure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import os
from PIL import Image
from resources import DTU_BASE_DIR
# DTU_BASE_DIR = "/home/jupyter/pixel-nerf/rs_dtu_4/DTU/"

DTU_TEST_WHITELIST = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]

# DEBUG=True
# if DEBUG:
#     print("debuggin!")
#     DTU_TEST_WHITELIST = [110]

def get_idx(scan_path):
    basename, _ = os.path.splitext(os.path.basename(scan_path))
    return int(basename)


def get_fov_deg(f,c):
    fov_rad = 2*np.arctan2(c, f)
    fov = np.rad2deg(fov_rad)
    return fov


def get_dtu_pose(all_cam, i):

    _coord_trans_world = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    )
    _coord_trans_cam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    )

    # https://github.com/sxyu/pixel-nerf/blob/91a044bdd62aebe0ed3a5685ca37cb8a9dc8e8ee/src/data/DVRDataset.py#L160
    P = all_cam["world_mat_" + str(i)]
    P = P[:3]

    K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    scale_mtx = all_cam.get("scale_mat_" + str(i))
    if scale_mtx is not None:
        norm_trans = scale_mtx[:3, 3:]
        norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

        pose[:3, 3:] -= norm_trans
        pose[:3, 3:] /= norm_scale

    pose = (
        _coord_trans_world
        @ pose
        @ _coord_trans_cam
    )
    world2cam = np.linalg.inv(pose)

    # get fov deg
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0, 2]
    cy = K[1, 2]

    fovx = get_fov_deg(fx, cx)
    fovy = get_fov_deg(fy, cy)
    fovx,fovy

    return world2cam, (fovx, fovy)


def get_camera_loc(polar_deg, azimuth_deg, radius_m):
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    # return to_sphere_theta_phi(azimuth_deg, -polar_deg + pi/2) * radius_m

    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)
    return np.array([cam_x, cam_y, cam_z])


def to_sphere_theta_phi(theta, phi):
    """Get coordinates on unit sphere given theta, phi.

    Args:
      theta: yaw angle in [0, 2pi)
      phi: pitch angle in (0, pi)
    Returns:
      coordinates on sphere in [-1, 1]^3
    """
    assert theta.ndim == 0
    assert phi.ndim == 0
    return np.array(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


def from_sphere_theta_phi(coords):
    x, y, z = coords
    theta = np.arctan2(y, x)
    theta = _get_positive_radians(theta)

    phi = np.arccos(z)
    return theta, phi


def _get_positive_radians(angle):
    angle = (angle + 2 * np.pi) % (2 * np.pi)
    return angle


def inv_get_camera_loc(xyz):
    # since get_camera_loc(phi, theta) == to_sphere_theta_phi(theta, phi + pi/2)
    # we have
    # inv_get_camera_loc(loc) == from_sphere_theta_phi(loc) - pi/2

    assert xyz.shape == (3,)
    radius = np.linalg.norm(xyz)
    xyz /= radius

    theta, phi = from_sphere_theta_phi(xyz)
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi - np.pi / 2)
    return phi, theta, radius

    # azimuth_rad = np.arctan2(y,x)
    # azimuth_rad = _get_positive_radians(azimuth_rad)
    # polar_rad = np.arccos(z) - np.pi/2
    # return np.rad2deg(polar_rad), np.rad2deg(azimuth_rad), radius

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    # camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
    #                       -np.sin(azimuth_rad),
    #                       -np.cos(azimuth_rad) * np.sin(polar_rad)],
    #                      [np.sin(azimuth_rad) * np.cos(polar_rad),
    #                       np.cos(azimuth_rad),
    #                       -np.sin(azimuth_rad) * np.sin(polar_rad)],
    #                      [np.sin(polar_rad),
    #                       0.0,
    #                       np.cos(polar_rad)]])
    # return camera_R

def get_dtu_scene(base_dir, scene_uid):
    scan = scene_uid
    scan_dir = os.path.join(base_dir, f"scan{scan}/image/")
    if not os.path.exists(scan_dir):
        print(f"scan_dir {scan_dir} does not exist, skipping")

    scan_paths = set(glob(os.path.join(scan_dir, "*.png")))

    cameras_path = os.path.join(base_dir, f"scan{scan}/cameras.npz")
    cameras = dict(np.load(cameras_path))

    # points = np.array(
    #     [
    #         get_dtu_pose(cameras, i)[:3, -1]
    #         for i in range(len(scan_paths))
    #     ]
    # )
    # mean_norm = np.sqrt((points**2).sum(axis=-1)).mean()

    # verify that intrinsics are constant
    all_intrinsics = np.array(
        [cameras[f"scale_mat_{i}"] for i in range(len(scan_paths))]
    )
    assert (all_intrinsics.std(axis=0).ravel() < 1e-8).all()

    idx_to_view = {}

    for scan_path in scan_paths:
        idx = get_idx(scan_path)
        pose, (fovx_deg, fovy_deg) = get_dtu_pose(cameras, idx)
        pose = np.linalg.inv(pose)

        uncropped_im_gt = Image.open(scan_path).convert("RGB")
        im_gt = common.central_crop_v2(uncropped_im_gt)

        im_gt = get_resized_np_img(im_gt, (256, 256))
        uncropped_im_gt = get_resized_np_img(uncropped_im_gt, (400, 300))
        idx_to_view[idx] = (im_gt,uncropped_im_gt, pose)

    true_idx_to_idx = np.array(sorted(idx_to_view))
    assert 25 in true_idx_to_idx

    views = [idx_to_view[idx] for idx in sorted(idx_to_view)]

    true_imgs = np.array([image for (image, _, _) in views])
    uncropped_true_imgs = np.array([image for (_,image, _) in views])
    true_poses = np.array([pose for (_, _,pose) in views])
    true_canonical_idx = true_idx_to_idx.tolist().index(25)

    return {
        "images": true_imgs,
        "uncropped_images": uncropped_true_imgs,
        "extrinsics": true_poses,
        "fovy_deg": fovy_deg,
        "test_input_idx": true_canonical_idx 
    }

        

def get_dtu_eval_pairs(base_dir):
    raise NotImplementedError
    # for scan in tqdm(DTU_TEST_WHITELIST):
        

    #     # main_intrinsics = all_intrinsics[0]
    #     # main_extrinsics = cameras[f'world_mat_inv_25']
    #     main_extrinsics = get_dtu_pose(cameras, 25)

    #     # if rescale:
    #     #     main_extrinsics[:3, -1] = main_extrinsics[:3, -1] / mean_norm

    #     src_scan_path = os.path.join(scan_dir, "000025.png")

    #     src_scan_im = Image.open(src_scan_path)
    #     src_scan_im = src_scan_im.convert("RGB")
    #     src_scan_im = common.central_crop_v2(src_scan_im)

    #     dst_scan_paths = set(scan_paths) - {src_scan_path}
    #     assert src_scan_path in scan_paths
    #     assert src_scan_path not in dst_scan_paths

    #     for dst_scan_path in dst_scan_paths:
    #         dst_im_gt = Image.open(dst_scan_path)
    #         dst_im_gt = common.central_crop_v2(dst_im_gt)

    #         idx = get_idx(dst_scan_path)

    #         novel_extrinsics = get_dtu_pose(cameras, idx)

    #         yield src_scan_im, main_extrinsics, dst_im_gt, novel_extrinsics


def get_resized_np_img(image, resolution):
    rh, rw = resolution
    im1 = image
    h1,w1 = im1.size
    # assert h1 == w1
    assert h1/w1 == rh/rw, (h1/w1, rh/rw)
    im1 = im1.resize((rh, rw), resample=Image.LANCZOS)
    im1_arr = np.array(im1)
    im1_arr = im1_arr / 255.
    im1_arr = im1_arr * 2 - 1
    assert im1_arr.shape == (rw, rh, 3)
    return im1_arr

# def get_serializable_np_batches(base_dir):
#     # as if it came from a preprocessed webdataset like in webdataset_utils
#     print("it's floated1")
#     resolution = 256

#     np_batches = []
#     for src_im, src_cam, dst_im, dst_cam in get_dtu_eval_pairs(base_dir):
#         src_im = get_resized_np_img(src_im, resolution)
#         dst_im = get_resized_np_img(dst_im, resolution)
#         assert dst_cam.shape == (4,4)
#         assert src_cam.shape == (4,4)

#         np_batch = {
#             "image_cond": src_im[None].astype(np.float32),
#             "image_target": dst_im[None].astype(np.float32),
#             'T': common.get_T(dst_cam[:3], src_cam[:3])[None].to(torch.float32),
#             'cond_cam2world': np.linalg.inv(src_cam)[None],
#             'target_cam2world': np.linalg.inv(dst_cam)[None],
#         }
#         np_batches.append(np_batch)

#     return np_batches