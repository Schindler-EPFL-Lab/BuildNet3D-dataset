import json
import logging
from pathlib import Path

import bpy

from buildnet3d_dataset.buildings.skyscraper import Skyscraper


class Annotation:
    """
    Class that writes an annotation based on Pix3D dataset structure from an
    active 3D scene.

    """

    def __init__(self) -> None:
        self.content = {}
        self.full = []
        self._clean()

    def add(self, building: Skyscraper, name: str, model: str) -> None:
        """
        Function that adds a model's annotation to the full dataset annotation.
        :param building: building to add to json, Building class
        :param name: name of the image file, str
        :param model: name of the model .obj file, str
        :return:
        """
        assert isinstance(name, str)

        self.content["img"] += name
        self.content["mask"] += name.split("/")[-1]
        self.content["point_cloud"] += name.split("/")[-1]
        self.content["model"] += model

        try:
            self.content["cam_position"] = list(
                bpy.data.objects["Camera"].location  # type: ignore
            )
            self.content["cam_position"] = [
                round(x, 3) for x in self.content["cam_position"]
            ]
        except Exception:
            pass
        try:
            self.content["focal_length"] = round(
                bpy.data.cameras["Camera"].lens,  # type: ignore
                3,  # type: ignore
            )

        except Exception:
            pass

        for v in building.volumes:
            try:
                self.content["material"].append(
                    v.mesh.active_material.name.split(".")[0]
                )
            except Exception:
                pass

        self.content["WWR"] = building.wwr
        self.content["material"] = list(set(self.content["material"]))
        self.content["img_size"] = (
            bpy.data.scenes[0].render.resolution_y,  # type: ignore
            bpy.data.scenes[0].render.resolution_x,  # type: ignore
        )
        self.content["bbox"] = building.get_bb()
        self.full.append(self.content)
        self._clean()

    def write(self, filepath: Path = Path("./test.json")) -> None:
        """
        Function that writes the full json annotation to the provided location.
        :param filepath: path for written file, pathlib.Path, default='./test.json'
        :return:
        """
        with filepath.open(mode="w") as f:
            json.dump(self.full, f)

        logging.info(f"Annotation successfully written as {str(filepath)}")

    def _clean(self) -> None:
        """
        Function that returns the annotation template to its default form.
        :return:
        """
        self.content = {
            "img": "images/",
            "category": "building",
            "img_size": None,
            "2d_keypoints": [],
            "mask": "masks/",
            "img_source": "synthetic",
            "model": "model/",
            "point_cloud": "pointCloud/",
            "model_raw": 0,
            "model_source": "synthetic",
            "trans_mat": 0,
            "focal_length": 35.0,
            "cam_position": (0.0, 0.0, 0.0),
            "rotation": (0, 0, 0),  # TODO: add camera rotation
            "inplane_rotation": 0,
            "truncated": False,
            "occluded": False,
            "slightly_occluded": False,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "material": [],
        }
