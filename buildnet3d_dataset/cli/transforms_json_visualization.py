"""
Visualize position of cameras from transforms.json

For each synthetic building, multi-views are generated. For each view, the position
of thecamera is recorded as a rotation (or transform) matrix in a file called
transforms.json (which contains also the parameters of the camera). This script
reads the file and plots the cameras.
"""
import json
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.camera as pc
import tyro
from pytransform3d.transform_manager import TransformManager


@dataclass
class Parameters:
    dataset_name: str
    """Name of the dataset where the building is"""
    building_number: int
    """Index of the building to plot"""
    data_path: pathlib.Path | None = None
    """Path to the folder where datasets are saved. Defaults to
    'path_to_repo/datasets'"""
    number_cameras_to_show: int = 100
    """Specify the maximum number of cameras to show if there are too many by default"""
    x_y_lim: float = 50
    """Plot x and y axis limit"""
    z_lim: float = 60
    """Plot z axis limit"""

    def __post_init__(self) -> None:
        if self.data_path is None:
            current_folder = pathlib.Path(__file__).parent.resolve()
            self.data_path = pathlib.Path(current_folder, "datasets")

        self.full_path = pathlib.Path(
            self.data_path,
            self.dataset_name,
            "images",
            "building" + str(self.building_number),
            "transforms.json",
        )
        if not self.full_path.is_file():
            raise FileNotFoundError(str(self.full_path) + " was not found")


def load_from_json(filename: pathlib.Path):
    """Load a dictionary from a JSON filename."""
    with open(str(filename), encoding="UTF-8") as file:
        return json.load(file)


def main() -> None:
    params = tyro.cli(Parameters)

    meta = load_from_json(params.full_path)
    poses = []

    frames = meta["frames"]
    for i, frame in enumerate(frames):
        if i < params.number_cameras_to_show:
            poses.append(
                np.array(frame["transform_matrix"])
                @ np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            )
        else:
            break

    # Setup the transform manager
    tm = TransformManager()
    for i, pose in enumerate(poses):
        tm.add_transform(str(i), "poses", pose)

    # default parameters of a camera in Blender
    sensor_size = np.array([meta["w"], meta["h"]])
    fl = meta["fl_x"]
    intrinsic_matrix = np.array(
        [[fl, 0, sensor_size[0] / 2.0], [0, fl, sensor_size[1] / 2.0], [0, 0, 1]]
    )
    virtual_image_distance = 6

    # Plot the transform's axis and the camera
    plt.figure(figsize=(30, 15))
    ax = tm.plot_frames_in("poses", s=0.1)
    for i, pose in enumerate(poses):
        pc.plot_camera(
            ax,
            cam2world=pose,
            M=intrinsic_matrix,
            sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance,
        )
    ax.set_xlim((-params.x_y_lim, params.x_y_lim))
    ax.set_ylim((-params.x_y_lim, params.x_y_lim))
    ax.set_zlim((0, params.z_lim))
    plt.show()


if __name__ == "__main__":
    main()
