import json
from pathlib import Path

import bpy
import numpy as np

from buildnet3d_dataset.blender_utils import deselect_all


class CameraManager:
    def __init__(self, path_to_dataset: Path, image_size: tuple[int, int]) -> None:
        """
        Class that manages the cameras in the scene and create the transforms.json

        Provide the `path_to_dataset` to store the transforms.json and the `image_size`
        which is necessary in the transforms.json
        """
        self.scene = bpy.context.scene  # type: ignore
        self.main_camera = bpy.data.objects["Camera"]  # type: ignore
        self.camera_infos = {}
        self.camera_infos["camera_model"] = "OPEN_CV"
        self.camera_infos["width"] = image_size[0]
        self.camera_infos["height"] = image_size[1]
        focal_length = (
            0.5 * image_size[0] / np.tan(0.5 * bpy.data.cameras["Camera"].angle_x)  # type: ignore
        )
        self.intrinsic = [
            [focal_length, 0, image_size[0] / 2, 0],
            [0, focal_length, image_size[1] / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        self.camera_infos["has_mono_prior"] = False
        self.camera_infos["has_foreground_mask"] = False
        self.camera_infos["has_sparse_sfm_points"] = False
        self.camera_infos["scene_box"] = {"aabb": [[-1, -1, -1], [1, 1, 1]]}
        self.camera_infos["frames"] = []

        self.path_to_dataset = path_to_dataset
        bpy.ops.object.camera_add()  # type: ignore
        self.camera = bpy.data.objects["Camera.001"]  # type: ignore
        _ = self._nest_camera()

    def _nest_camera(self) -> bpy.types.Collection:  # type: ignore
        deselect_all()
        bpy.data.collections["Collection"].objects.link(self.camera)  # type: ignore
        deselect_all()
        self.camera.select_set(True)
        bpy.ops.collection.objects_remove(collection="Building")  # type: ignore
        return bpy.data.collections["Collection"]  # type: ignore

    def make_main(self) -> None:
        """
        Function that changes the camera to the main one.
        """
        self.scene.camera = self.main_camera

    def make(self) -> None:
        """
        Function that changes the camera to the secondary one and sets its orientation.
        """

        self.scene.camera = self.camera

        self.camera.rotation_euler[0] = np.radians(
            np.random.randint(40, 100) + np.random.random()
        )
        self.camera.rotation_euler[2] = np.radians(
            np.random.randint(0, 360) + np.random.random()
        )

        bpy.context.view_layer.update()  # type:ignore

    def make_specific(self, rotation_euler: tuple[float, float, float]) -> None:
        """
        Rotates camera so that it is pointing in the direction specified by
        `rotation_euler`, which is a length 3 tuple containing rotation in degrees.
        Camera is rotated from pointing downwards.
        """
        self.scene.camera = self.camera

        for i, rotation in enumerate(rotation_euler):
            self.camera.rotation_euler[i] = np.radians(rotation)

        bpy.context.view_layer.update()  # type:ignore

    def record_camera(self, building_index: int, view_index: int) -> None:
        self.camera_infos["frames"].append(
            {
                "rgb_path": "building_{}_{}.png".format(building_index, view_index),
                "segmentation_path": "building_{}_{}_mask.png".format(
                    building_index, view_index
                ),
                "camtoworld": np.array(self.camera.matrix_world).tolist(),
                "intrinsics": self.intrinsic,
            }
        )

    def write_camera_infos(self, building_index: int) -> None:
        json_path = Path(
            self.path_to_dataset, f"images/building{building_index}/meta_data.json"
        )
        with json_path.open(mode="w") as outfile:
            json.dump(self.camera_infos, outfile, indent=4)

    def reset_frames_information(self) -> None:
        """Used to reset frames when starting a new building"""
        self.camera_infos["frames"] = []
