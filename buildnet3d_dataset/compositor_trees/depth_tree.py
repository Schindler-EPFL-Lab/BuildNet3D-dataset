from pathlib import Path

from bpy.types import CompositorNode  # type: ignore

from buildnet3d_dataset.compositor_trees.tree import Tree


class DepthTree(Tree):
    def __init__(self, base_path: Path) -> None:
        """
        `base_path` is the path into the dataset output directory.
        """
        super().__init__(base_path=Path(base_path, "depths"), suffix="depth")
        self._file_ouput_node.name = "Depth Output"
        self._file_ouput_node.format.file_format = "HDR"

    def _make_full_tree(self) -> tuple[CompositorNode, str]:  # type: ignore
        """
        Function that creates a node tree outputs the depth returned from the
        "Render Layer" node.
        """
        return self._root_node, "Depth"
