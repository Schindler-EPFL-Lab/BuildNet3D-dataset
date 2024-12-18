import pathlib
from dataclasses import dataclass

import tyro
from open3d import io, visualization


@dataclass
class DataSelection:
    dataset_name: str
    """name of the dataset where the building is"""
    building_number: int
    """Index of the building to plot"""
    data_path: pathlib.Path | None = None
    """Path to the folder where datasets are saved. Defaults to
    'path_to_repo/dataset'"""

    def __post_init__(self) -> None:
        if self.data_path is None:
            current_folder = pathlib.Path(__file__).parent.resolve()
            self.data_path = pathlib.Path(current_folder, "dataset")

        self.full_path = pathlib.Path(
            self.data_path,
            self.dataset_name,
            "PointCloud",
            str(self.building_number) + ".ply",
        )
        if not self.full_path.is_file():
            raise FileNotFoundError(str(self.full_path) + " was not found")


def main() -> None:
    data_selection = tyro.cli(DataSelection)

    cloud = io.read_point_cloud(str(data_selection.full_path))
    visualization.draw_geometries([cloud])  # type: ignore


if __name__ == "__main__":
    main()
