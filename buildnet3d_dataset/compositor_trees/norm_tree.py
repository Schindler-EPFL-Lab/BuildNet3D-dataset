from pathlib import Path

from bpy.types import CompositorNode  # type: ignore

from buildnet3d_dataset.compositor_trees.tree import Tree


class NormTree(Tree):
    def __init__(self, base_path: Path) -> None:
        """
        `base_path` is the path into the dataset output directory.
        """
        super().__init__(base_path=Path(base_path, "norms"), suffix="norm")
        self._file_ouput_node.name = "Normals Output"

    def _make_full_tree(self) -> tuple[CompositorNode, str]:  # type: ignore
        """
        Function that returns the "Render Layer" (`self._root_node`) and the output
        key to access the normals data.
        """
        return self._root_node, "Normal"
