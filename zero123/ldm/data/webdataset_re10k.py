from ldm.data import webdataset_co3d
import numpy as np
import json
import io


def get_fov_deg(*, camtopix):
    fx = camtopix[0, 0]
    fy = camtopix[1, 1]
    cx = camtopix[0, 2]
    cy = camtopix[1, 2]

    # import pdb
    # pdb.set_trace()

    assert np.abs(fx - fy) < 1e-5  # must be square

    fov = 2 * np.arctan2(cx, fx)
    fov_deg = fov * 180 / np.pi
    return fov_deg


class RE10K(webdataset_co3d.CO3D):
    def load_sample_metadata(self, metadata_bytes):
        metadata = json.load(io.BytesIO(metadata_bytes))

        extrinsics = np.array(metadata["extrinsics"])
        worldtocam = np.linalg.inv(extrinsics)[:3]
        assert worldtocam.shape == (3, 4)

        return worldtocam, metadata

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

        world_data = webdataset_co3d.get_world_data(worldtocams=cams)

        metadata = next(iter(scene["metadatas"].values()))
        pixtocam = np.array(metadata["intrinsics"])
        camtopix = np.linalg.inv(pixtocam)

        fov_deg = get_fov_deg(camtopix=camtopix)

        camera_data = {"fov_deg": fov_deg}

        metadata = {"true_idx_to_idx": true_idx_to_idx}
        return camera_data, world_data, metadata, scene
