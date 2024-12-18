import numpy as np

from buildnet3d_dataset.buildings.composed_building import ComposedBuilding
from buildnet3d_dataset.buildings.volume import Volume


class Skyscraper(ComposedBuilding):
    def __init__(
        self,
        semantic_id: int,
        min_width: float,
        min_length: float,
        min_height: float,
        max_width: float,
        max_length: float,
        max_height: float,
        segmentation_color: tuple[float, float, float],
        template: bool = False,
    ) -> None:
        """
        Class that represents a Skyscraper building with height significantly larger
        than width or length of the building. `semantic_id` and `segmentation_color`
        are used for semantic segmentation. If template is `True` the building is not
        generated in blender and only created as a python representation.

        The size is chosen randomly in the closed interval [`min_width`, `max_width`].
        [`min_length`, `max_length`], and [`min_height`, `max_height`]
        """
        self.min_width = min_width
        self.min_length = min_length
        self.min_height = min_height
        self.max_width = max_width
        self.max_length = max_length
        self.max_height = max_height
        self.width = np.random.randint(int(self.min_width), int(self.max_width + 1))
        self.length = np.random.randint(int(self.min_length), int(self.max_length + 1))
        self.height = np.random.randint(int(self.min_height), int(self.max_height + 1))
        super().__init__(
            semantic_id=semantic_id,
            segmentation_color=segmentation_color,
            template=template,
        )

    def _generate(self) -> None:
        v = Volume(
            id=self.semantic_id,
            name="building_block",
            width=self.width,
            length=self.length,
            height=self.height,
            segmentation_color=self.segmentation_color,
        )
        self.volumes.append(v)

    @property
    def wwr(self) -> float:
        """
        Extract windows area and wall area to compute the Windows-to-Wall Ratio
        """
        if self._windows_applied == 0:
            return 0

        building_width = self.volumes[0].width
        building_length = self.volumes[0].length
        building_height = self.volumes[0].height
        windows_area = self._windows_area

        wall_area = (
            2 * building_width * building_height
            + 2 * building_length * building_height
            + building_width * building_length
        )
        return windows_area / wall_area
