import numpy as np

from buildnet3d_dataset.blender_utils import gancio
from buildnet3d_dataset.buildings.composed_building import ComposedBuilding
from buildnet3d_dataset.buildings.volume import Volume


class LBuilding(ComposedBuilding):
    def __init__(
        self,
        semantic_id: int,
        min_width: int,
        min_length: int,
        min_height: int,
        min_cutout_width_percent: float,
        min_cutout_length_percent: float,
        max_width: int,
        max_length: int,
        max_height: int,
        max_cutout_width_percent: float,
        max_cutout_length_percent: float,
        segmentation_color: tuple[float, float, float],
        template: bool = False,
    ) -> None:
        """
        Class that represents a a building in the shape of an "L" when looked from
        overhead. `semantic_id` and `segmentation_color` are used for semantic
        segmentation. If template is `True` the building is not generated in blender
        and only created as a python representation.

        The bounding box of the building is chosen between the ranges of
        [`min_width`, `max_width`], [`min_length`, `max_length`] and
        [`min_height`, `max_height`]. The L shape is formed by cutting out a corner of
        the bounding box. The size of the cutout is determined as a percentage of
        the building width and length in the range of
        [`min_cutout_width_percent`, `max_cutout_width_percent`] by
        [`min_cutout_length_percent`, `max_cutout_length_percent`].
        """
        self._min_width = min_width
        self._min_length = min_length
        self._min_height = min_height
        self._min_cutout_width_percent = min_cutout_width_percent
        self._min_cutout_length_percent = min_cutout_length_percent
        self._max_width = max_width
        self._max_length = max_length
        self._max_height = max_height
        self._max_cutout_width_percent = max_cutout_width_percent
        self._max_cutout_length_percent = max_cutout_length_percent
        self._width = np.random.randint(int(self._min_width), int(self._max_width) + 1)
        self._length = np.random.randint(
            int(self._min_length), int(self._max_length) + 1
        )
        self._height = np.random.randint(
            int(self._min_height), int(self._max_height) + 1
        )
        self._cutout_width_perc = np.random.uniform(
            low=self._min_cutout_width_percent, high=self._max_cutout_width_percent
        )
        self._cutout_length_perc = np.random.uniform(
            low=self._min_cutout_length_percent, high=self._max_cutout_length_percent
        )
        self._cutout_width = int(np.floor(self._width * self._cutout_width_perc))
        self._cutout_length = int(np.floor(self._length * self._cutout_length_perc))

        super().__init__(
            semantic_id=semantic_id,
            segmentation_color=segmentation_color,
            template=template,
        )

    def _generate(self) -> None:
        """
        Generates and attaches volumes of the L-shaped building.
        """
        vol_1_length = self._length
        vol_1_width = self._width - self._cutout_width
        vol_1 = Volume(
            id=self.semantic_id,
            name="volume",
            width=vol_1_width,
            length=vol_1_length,
            height=self._height,
            segmentation_color=self.segmentation_color,
        )
        self.volumes.append(vol_1)

        vol_2_length = self._length - self._cutout_length
        vol_2_width = self._cutout_width + self._overlap_distance
        vol_2 = Volume(
            id=self.semantic_id,
            name="volume",
            width=vol_2_width,
            length=vol_2_length,
            height=self._height,
            segmentation_color=self.segmentation_color,
        )
        self.volumes.append(vol_2)

        gancio(
            vol_1, vol_2, axis=0, border1=0, border2=1, overlap=self._overlap_distance
        )

    @property
    def wwr(self) -> float:
        """
        Extract windows area and wall area to compute the Windows-to-Wall Ratio
        """
        if self._windows_applied == 0:
            return 0

        windows_area = self._windows_area

        wall_area = (
            2 * self._width * self._height
            + 2 * self._length * self._height
            + self._width * self._length
            - self._cutout_width * self._cutout_length
        )
        return windows_area / wall_area
