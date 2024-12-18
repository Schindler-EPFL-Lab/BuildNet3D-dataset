from buildnet3d_dataset.buildings.volume import Volume


class Roof(Volume):
    def __init__(
        self,
        semantic_id: int,
        width: float,
        length: float,
        thickness: float,
        location: tuple[float, float, float],
        segmentation_color: tuple[float, float, float],
    ) -> None:
        """
        Creates a Roof object with dimensions `width`, `length` and `thickness` at
        `location`. `semantic_id` and `segmentation_color` are used for semantic
        segmentation.
        """
        super().__init__(
            id=semantic_id,
            name="roof",
            width=width,
            length=length,
            height=thickness,
            location=location,
            floor_height=0,
            segmentation_color=segmentation_color,
        )

    @classmethod
    def from_volume(
        cls,
        semantic_id: int,
        segmentation_color: tuple[float, float, float],
        volume: Volume,
        overhang: float = 1,
        thickness: float = 0.5,
    ) -> "Roof":
        """
        Creates a `Roof` object and places it centered on the `volume`. Roof object is
        created so that it is larger than volume by `overhang` amount on all sides of
        the volume and `thickness` tall. `semantic_id` and `segmentation_color` are
        used for semantic segmentation.
        """
        width = volume.width + overhang * 2
        length = volume.length + overhang * 2
        volume_location = volume.mesh.location
        height = volume.height
        return cls(
            semantic_id=semantic_id,
            width=width,
            length=length,
            thickness=thickness,
            location=(volume_location[0], volume_location[1], height),
            segmentation_color=segmentation_color,
        )
