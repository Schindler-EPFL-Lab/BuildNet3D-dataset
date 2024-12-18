from unittest.mock import MagicMock

from buildnet3d_dataset.buildings.composed_building import ComposedBuilding
from buildnet3d_dataset.buildings.volume import Volume


class MockComposedBuilding(ComposedBuilding):
    """
    Creates a Mock Composed Building with 1 volume of width, length, height of 15, 5,
    7.
    """

    def __init__(
        self, semantic_id=1, segmentation_color=(0.5, 0.5, 0.5), template=False
    ) -> None:
        super().__init__(
            semantic_id=semantic_id,
            segmentation_color=segmentation_color,
            template=template,
        )

    def _generate(self) -> None:
        vol = Volume(id=0, name="volume", width=15, length=5, height=7)
        self.volumes.append(vol)

    def wwr(self):
        return MagicMock()
