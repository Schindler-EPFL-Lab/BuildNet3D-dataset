import os

import numpy as np

from buildnet3d_dataset.material import (
    GlassMaterial,
    MaskMaterial,
    Material,
    MetallMaterial,
)


class MaterialFactory:
    """
    Class that produces materials based on the given name.
    """

    def __init__(self, file_dir: str) -> None:
        self.file_dir = file_dir
        self.materials = os.listdir("{}/Textures".format(self.file_dir))
        self.materials = [
            x
            for x in self.materials
            if os.path.isdir("{}/Textures/{}".format(self.file_dir, x))
        ]

    def produce(
        self, name: str | None = None, color: tuple[float, float, float] | None = None
    ) -> Material:
        if name:
            if name == "mask":
                if not color:
                    color = (1.0, 0.0, 0.0)
                return MaskMaterial(self.file_dir, name, color=color)
            elif name == "glass":
                return GlassMaterial(self.file_dir, name.lower().capitalize())
            elif name == "metall":
                return MetallMaterial(self.file_dir, name.lower().capitalize())
            else:
                name = name.lower().capitalize()
                assert (
                    name in self.materials
                ), "Unknown material {}, not in Textures folder".format(name)
                return Material(name, self.file_dir)
        else:
            return Material(np.random.choice(self.materials), self.file_dir)
