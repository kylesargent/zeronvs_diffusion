import numpy as np
import visu3d as v3d
import math
from PIL import Image


def fi(t, i=0):
    if t.ndim == 4:
        t = t[i]
    t = np.array(t)  # .transpose((1,2,0))
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(t)


def get_camera_matrices_as_rays(camera_matrices):
    _, camera_extrinsics = camera_matrices
    x, y, z = camera_extrinsics[:, :3, :3].transpose((2, 0, 1))
    loc = camera_extrinsics[:, :3, -1]

    x_ray = v3d.Ray(pos=loc, dir=x)
    y_ray = v3d.Ray(pos=loc, dir=y)
    z_ray = v3d.Ray(pos=loc, dir=z)

    return [x_ray, y_ray, z_ray]


def get_homogeneous_camtoworld(*, inhomogeneous_worldtocam):
    square_cams = np.array(
        [
            np.concatenate([cam, np.array([0, 0, 0, 1])[None]], axis=0)
            for cam in inhomogeneous_worldtocam
        ]
    )
    square_cams = np.linalg.inv(square_cams)
    return square_cams


def visualize_cameras(*, inhomogeneous_worldtocam):
    #  This codebase assumes a [3,4] worldtocam representation with
    #  x = right
    #  y = up
    #  z = toward the viewer
    #  If the output of this function doesn't show that, you have a data bug!

    #  To visualize the cameras we need to invert to get the cam2world
    #  representation so that the [:, :3, -1] slice represents the camera locations.
    square_cams = get_homogeneous_camtoworld(
        inhomogeneous_worldtocam=inhomogeneous_worldtocam
    )

    cam_rays = get_camera_matrices_as_rays((None, np.array(square_cams)))
    return cam_rays


matmul = np.matmul
mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]


def get_canonical_cameras():
    intrinsics = np.array(
        [
            [0.00357107, 0.0, -0.45709702, 0.0],
            [0.0, 0.00357107, -0.45709702, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsics = np.array(
        [
            [0.0, 0.0, -1.0, -2.732],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, extrinsics


def get_homogeneous_intrinsics(intrinsics):
    assert intrinsics.shape == (3, 3)
    homogeneous_intrinsics = np.eye(4)
    homogeneous_intrinsics[:3, :3] = intrinsics
    return homogeneous_intrinsics


def get_inhomogeneous_intrinsics(intrinsics):
    assert intrinsics.shape == (4, 4)
    return intrinsics[:3, :3]


def get_homogeneous_extrinsics(extrinsics):
    assert extrinsics.shape == (3, 4)
    return np.concatenate([extrinsics, np.array([[0, 0, 0, 1]])], axis=0)


def get_inhomogeneous_extrinsics(extrinsics):
    assert extrinsics.shape == (4, 4)
    return extrinsics[:3]


def get_normals(depth, intrinsics, extrinsics):
    focal = np.linalg.inv(intrinsics)[0, 0]
    # focal length in pixels

    assert depth.ndim == 3
    assert depth.shape[2] == 1
    depth = np.clip(depth[..., 0], a_min=1e-3)

    dz_dv, dz_du = np.gradient(depth)
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = focal / depth  # x is xyz of camera coordinate
    dv_dy = focal / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    # normal /= np.clip(n, 1e-3)
    normal /= n
    # normal = jax.vmap(lambda x: np.pad(x, 1, 'symmetric'), -1, -1)(
    #     normal[1:-1, 1:-1]
    # )
    # normal = normal[..., ::-1]
    normal = normal.reshape((-1, 3, 1))
    # normal = normal[..., 0]
    transform = np.matmul(
        extrinsics[:3, :3],
        np.diag(np.array([-1, 1, 1])),
    )
    normal = jax.vmap(np.matmul, (None, 0))(transform, normal)[..., 0]
    return normal


def pixel_coords(w, h):
    return np.meshgrid(np.arange(w), np.arange(h), indexing="xy")


def get_homogeneous_point_cloud(image, depth, intrinsics, extrinsics):
    h, w, three = image.shape
    assert h == w
    assert three == 3
    assert depth.shape == (h, w, 1)
    assert intrinsics.shape == (4, 4)
    assert extrinsics.shape == (4, 4)

    n = h * w

    ## first, image, depth -> point cloud
    pixel_x, pixel_y = pixel_coords(h, w)
    pixelspace_locations = np.stack(
        [pixel_x + 0.5, pixel_y + 0.5, np.ones_like(pixel_x)], axis=-1
    )
    pixelspace_locations = pixelspace_locations.reshape((-1, 3))
    color = image.reshape((-1, 3))
    assert color.shape == (n, 3)

    pixtocam = intrinsics[:3, :3]
    camtoworld = extrinsics

    cameraspace_locations = mat_vec_mul(pixtocam, pixelspace_locations)
    cameraspace_locations = matmul(
        cameraspace_locations, np.diag(np.array([1.0, -1.0, -1.0]))
    )
    cameraspace_locations *= depth.reshape((-1, 1))

    homogeneous_cameraspace_locations = np.concatenate(
        [cameraspace_locations, np.ones_like(cameraspace_locations[..., :1])],
        axis=-1,
    )

    homogeneous_worldspace_locations = mat_vec_mul(
        camtoworld, homogeneous_cameraspace_locations
    )

    return homogeneous_worldspace_locations, color


def visualize_homogeneous_point_cloud(homogeneous_worldspace_locations, color, clip=20):
    return v3d.Point3d(
        p=np.clip(homogeneous_worldspace_locations[..., :3], -clip, clip),
        rgb=np.clip(color * 255, 0, 255).astype(np.uint8),
    )


def get_fov_deg(intrinsics):
    pixtocam = intrinsics[:3, :3]
    camtopix = np.linalg.inv(pixtocam)
    focal, _, half_plane = camtopix[0]
    return 2 * np.arctan2(half_plane, focal) * 180 / np.pi
