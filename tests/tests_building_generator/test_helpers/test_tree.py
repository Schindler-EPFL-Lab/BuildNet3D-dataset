from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from buildnet3d_dataset.compositor_trees.tree import Tree


class MockTree(Tree):
    """
    Used to instantiate the `Tree` class for testing.
    """

    def __init__(self, base_path: Path, suffix: str) -> None:
        """
        `base_path` is the path to where images from the Compositor tree pipline will
        be ouput. The 'sub_path' property is for any additional folder structuring from
        the base path as well as the name of the output file general output file
        without extension. `suffix` is additional information to be added to the end of
        the file name, again without the file extension. The extension will be added by
        blender automatically depending on the properties of the FileOutput Node.
        """
        super().__init__(base_path, suffix)

    def _make_full_tree(self) -> tuple[Any, str]:
        return MagicMock(), "test"
