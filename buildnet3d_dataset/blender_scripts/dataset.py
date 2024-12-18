import argparse
import itertools
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from time import time

import jsonpickle

from buildnet3d_dataset.annotation import Annotation
from buildnet3d_dataset.buildings.composed_building import (
    ComposedBuilding,
)
from buildnet3d_dataset.cameramanager import (
    CameraManager,
)
from buildnet3d_dataset.light_manager.uniform_light import (
    UniformLight,
)
from buildnet3d_dataset.material_factory import (
    MaterialFactory,
)
from buildnet3d_dataset.mesh_tools.segmented_3D_model import (
    Segmented3DModel,
)
from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)
from buildnet3d_dataset.renderer import Renderer


class Dataset:
    def __init__(
        self,
        output_base_path: str,
        segmentation_mapping: dict[str, dict[str, int | tuple[float, float, float]]],
        material_texture_path: Path,
        size: int,
        render_views: int,
        roof_material: list[str],
        window_material: list[str],
        buildings: list[ComposedBuilding],
        image_size: tuple[int, int] = (500, 500),
        specific_views: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """
        `output_base_path` is the path where all of the outputs of the dataset creation
        will be saved to.
        """
        self._name = "Building_dataset_{}_{}_{}".format(
            datetime.now().year, datetime.now().month, datetime.now().day
        )
        self._size = size

        self._render_views = render_views

        self._image_size = image_size
        self._output_base_path = output_base_path
        self._json = Annotation()

        self._material_texture_path = material_texture_path
        self._material_factory = MaterialFactory(str(material_texture_path))

        self._buildings = buildings

        self._number_of_modules = len(segmentation_mapping) - 2
        self._segmentation_mapping = segmentation_mapping

        self._building_id: int = self._segmentation_mapping["wall"][  # type: ignore
            "ID"
        ]
        self._building_rgb: tuple[float, float, float] = self._segmentation_mapping[
            "wall"
        ]["RGB"]  # type: ignore

        self._roof_id: int = self._segmentation_mapping["roof"]["ID"]  # type: ignore
        self._roof_rgb: tuple[float, float, float] = self._segmentation_mapping["roof"][
            "RGB"
        ]  # type: ignore
        self._roof_material: list[str] = roof_material

        self._window_id: int = self._segmentation_mapping["window"][  # type: ignore
            "ID"
        ]
        self._window_rgb: tuple[float, float, float] = self._segmentation_mapping[
            "window"
        ]["RGB"]  # type: ignore
        self._window_material: list[str] = window_material
        self._specific_views = specific_views

    def _write_segmentation_infos(
        self,
        building_index: int,
    ) -> None:
        colors = [x["RGB"] for x in self._segmentation_mapping.values()]
        segmentation_infos = {
            index: tuple(color)
            for index, color in enumerate(colors)  # type: ignore
        }
        json_path = Path(
            self._output_base_path,
            f"images/building{building_index}/segmentation_data.json",
        )
        with json_path.open(mode="w") as outfile:
            json.dump(segmentation_infos, outfile, indent=4)

    def populate(self) -> None:
        """
        Function that creates the dataset samples.
        """
        s = time()
        renderer = Renderer(
            Path(self._output_base_path),
            self._image_size,
            nbr_modules=self._number_of_modules,
            mandatory_class_color={
                x["ID"]: x["RGB"]
                for x in self._segmentation_mapping.values()  # type: ignore
            },
        )
        cameramanager = CameraManager(Path(self._output_base_path), self._image_size)
        for i in range(self._size):
            cameramanager.make()
            uniform_light = UniformLight()
            uniform_light.make()
            idx = random.randint(0, len(self._buildings) - 1)
            building = self._buildings[idx].clone(
                semantic_id=self._building_id, segmentation_color=self._building_rgb
            )

            building.add_roof(
                roof_id=self._roof_id, roof_segmentation_color=self._roof_rgb
            )
            building_material = self._material_factory.produce()
            roof_material = self._material_factory.produce(
                random.choice(self._roof_material)
            )

            building.apply_materials(
                building_material=building_material, roof_material=roof_material
            )
            building.add_window_elements(
                module=ParametricWindow(
                    id=self._window_id,
                    segmentation_color=self._window_rgb,
                    material=self._window_material,
                    path_to_materials=str(self._material_texture_path),
                ),
                x_step=random.randint(2, 6),
            )

            self._json.add(
                building,
                "{}.png".format(i),
                "{}.obj".format(i),  # type: ignore
            )

            building.remove_overlap_faces()

            view_id = 0
            if self._specific_views is not None:
                for view_angle in self._specific_views:
                    cameramanager.make_specific(view_angle)
                    renderer.render(
                        sub_path=Path(f"building{i}/building_{i}_{view_id}")
                    )
                    cameramanager.record_camera(i, view_id)
                    view_id += 1

            for _ in range(self._render_views):
                cameramanager.make()

                renderer.render(sub_path=Path(f"building{i}/building_{i}_{view_id}"))
                cameramanager.record_camera(i, view_id)
                view_id += 1

            building.clean_and_triangulate()

            building.save_obj(i, self._output_base_path)
            building.save_ply(i, self._output_base_path)
            building.save_stl(i, self._output_base_path)
            building.save_dae(i, self._output_base_path)
            building.save_segmented_mesh(i, self._output_base_path)
            cameramanager.write_camera_infos(i)
            cameramanager.reset_frames_information()
            self._write_segmentation_infos(
                building_index=i,
            )
            building.demolish()

        logging.info("Whole process took: {}".format(time() - s))

    def write(self) -> None:
        """
        Function that writes a json annotation to the dataset.
        """
        self._json.write(Path(self._output_base_path, f"{self._name}.json"))
        return

    def write_wall_wwr_information(self, segmentation_mapping_path: Path) -> None:
        logging.info("Starting writing of wall WWR...")
        ply_model_folder = Path(self._output_base_path, "model")
        json_output_path = Path(self._output_base_path, "wwr_information")
        json_output_path.mkdir(exist_ok=True, parents=True)
        for file in ply_model_folder.iterdir():
            if file.suffix == ".ply":
                logging.info(f"Starting building model: {file.stem}{file.suffix}")
                building = Segmented3DModel(
                    model_path=file, segmentation_mapping_path=segmentation_mapping_path
                )
                filename = file.stem.replace("_segmented", "")
                building.write_out_building_wall_wwr(
                    output_dir=json_output_path, filename=f"building{filename}_wwr"
                )

    @staticmethod
    def _read_segmentation_info(
        seg_mapping_path,
    ) -> dict[str, dict[str, int | tuple[float, float, float]]]:
        with open(seg_mapping_path, "r") as f:
            seg_mapping = json.load(f)
        return_mapping: dict[str, dict[str, int | tuple[float, float, float]]] = {}
        id_set: set[int] = set()
        total_num_ids = 0
        for key_1, val_dict in seg_mapping.items():
            return_mapping[key_1] = {}
            class_id = val_dict["ID"]
            class_rgb = val_dict["RGB"]

            assert isinstance(class_id, int), (
                f"Invalid ID: '{class_id}' set for '{key_1}' class. ID must be an "
                + "integer."
            )
            id_set.add(class_id)
            total_num_ids += 1
            return_mapping[key_1]["ID"] = class_id

            assert isinstance(class_rgb, tuple) | isinstance(class_rgb, list), (
                f"Invalid RGB code: '{class_rgb}' set for '{key_1}' class."
                + " RGB code must be tuple or list."
            )
            assert len(class_rgb) == 3, (
                f"Invalid RGB code: '{class_rgb}' set for '{key_1}' class."
                + "RGB code must have three values."
            )
            for i in range(3):
                assert isinstance(class_rgb[i], int), (
                    f"Invalid RGB code: '{class_rgb}' set for '{key_1}' class."
                    + "RGB code must have three integer values."
                )

            return_mapping[key_1]["RGB"] = (
                seg_mapping[key_1]["RGB"][0] / 255,
                seg_mapping[key_1]["RGB"][1] / 255,
                seg_mapping[key_1]["RGB"][2] / 255,
            )
        use_set = set(["ID", "RGB"])
        key_set = set(
            itertools.chain.from_iterable(
                [seg_class.keys() for seg_class in seg_mapping.values()]
            )
        )
        unsed_keys = key_set - use_set
        for key in unsed_keys:
            logging.warning(
                f"Information '{key}' from file"
                + f" {seg_mapping_path} is not currently implemented in"
                + " dataset generation"
            )
        assert (
            len(id_set) == total_num_ids
        ), "Invalid class IDs. All classes must have unique IDs."
        return return_mapping

    @classmethod
    def from_file(
        cls,
        dataset_config_path: Path,
        segmentation_mapping_path: Path,
        building_config_path: Path,
        material_texture_path: Path,
        output_base_path: str | None = None,
    ) -> "Dataset":
        """
        Function to create a dataset object from several configuration files.

        `dataset_config_path` is the path to the `.json` file that contains the
        information required to make a Datasetclass other than the
        `segmentation_mapping`, `building_param_dict`, and `material_texture_path`.
        `segmentation_mapping_path` is the path to the `.json` file that contains the
        information for semantic segmentation of the building components.
        `building_config_path` is the path to the `.json` tile that contains all
        all required inputs to generate any building in the `BuildingFactory` class.
        `material_texture_path` is the path to the directory that contains the
        `Textures/` folder which contains the texture information. `output_base_path`
        is the path that the outputs should be saved to, if provided it overrides the
        output path provided in the dataset configuration json file.
        """
        segmentation_mapping = cls._read_segmentation_info(segmentation_mapping_path)
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)
        buildings: list[ComposedBuilding] = []
        for file in building_config_path.iterdir():
            with open(file, "r") as f:
                json_string = json.load(f)
                building = jsonpickle.decode(json_string)
                if not issubclass(type(building), ComposedBuilding):
                    raise TypeError(
                        f"json decoded at {file} is not a subclase of ComposedBuilding"
                    )
                buildings.append(building)  # type: ignore
        if output_base_path is not None:
            dataset_config["output_base_path"] = output_base_path
        dataset = cls(
            segmentation_mapping=segmentation_mapping,
            material_texture_path=material_texture_path,
            buildings=buildings,
            **dataset_config,
        )
        return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    if "--" not in sys.argv:
        raise Exception("Missing '--' break in command")
    args = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])[0]

    output_dir = str(Path(Path(__file__).parents[1], "datasets/"))
    if args.output_dir is not None:
        output_dir = args.output_dir

    config_base_path = Path(Path(__file__).parents[1], "config")

    d = Dataset.from_file(
        output_base_path=output_dir,
        segmentation_mapping_path=Path(config_base_path, "segmentation_config.json"),
        dataset_config_path=Path(config_base_path, "dataset_config.json"),
        building_config_path=Path(config_base_path, "building_configs"),
        material_texture_path=config_base_path,
    )
    d.populate()
    d.write()
    d.write_wall_wwr_information(
        segmentation_mapping_path=Path(config_base_path, "segmentation_config.json")
    )


if __name__ == "__main__":
    main()
