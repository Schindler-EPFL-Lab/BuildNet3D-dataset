import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from buildnet3d_dataset.compositor_trees.depth_tree import DepthTree
from buildnet3d_dataset.compositor_trees.mask_node_tree import MaskNodeTree
from buildnet3d_dataset.compositor_trees.norm_tree import NormTree
from tests.tests_building_generator.test_helpers.test_tree import MockTree


class TestCompositorTrees(unittest.TestCase):
    @patch("bpy.data")
    def test_tree_sub_path(self, mock_data) -> None:
        suffix = "test-suffix"
        base_path = Path("test/base/path")
        sub_path = Path("test/sub/path")
        test_tree = MockTree(base_path=base_path, suffix=suffix)
        test_tree.sub_path = sub_path
        assert test_tree.sub_path == Path("test/sub/path_test-suffix")

    @patch("bpy.data")
    def test_tree_full_path(self, mock_data) -> None:
        suffix = "test-suffix"
        base_path = Path("test/base/path")
        sub_path = Path("test/sub/path")
        test_tree = MockTree(base_path=base_path, suffix=suffix)
        test_tree.sub_path = sub_path
        test_tree._file_format = "png"
        assert test_tree.full_path == Path(
            "test/base/path/test/sub/path_test-suffix0001.png"
        )

    @patch("bpy.data")
    def test_mask_node_tree_base_path(self, mock_data) -> None:
        base_path = Path("test/base/path")
        nbr_modules = 4
        mock_class_color_map = MagicMock()
        test_tree = MaskNodeTree(
            base_path=base_path,
            nbr_modules=nbr_modules,
            class_color_map=mock_class_color_map,
        )
        assert test_tree.base_path == Path("test/base/path/masks")

    @patch("bpy.data")
    def test_norm_tree_base_path(self, mock_data) -> None:
        base_path = Path("test/base/path")
        test_tree = NormTree(base_path=base_path)
        assert test_tree.base_path == Path("test/base/path/norms")

    @patch("bpy.data")
    def test_depth_tree_base_path(self, mock_data) -> None:
        base_path = Path("test/base/path")
        test_tree = DepthTree(base_path=base_path)
        assert test_tree.base_path == Path("test/base/path/depths")
