from abc import ABC, abstractmethod
from pathlib import Path

import bpy
from bpy.types import CompositorNode  # type: ignore


class Tree(ABC):
    """
    Generic compositor tree.
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
        self._scene = bpy.data.scenes[0]  # type: ignore
        self._scene.use_nodes = True
        self._links = self._scene.node_tree.links
        self._root_node = self._scene.node_tree.nodes["Render Layers"]
        self._margin = 60
        self._results_node, self._results_node_output_key = self._make_full_tree()
        self._file_ouput_node = self._scene.node_tree.nodes.new(
            "CompositorNodeOutputFile"
        )
        self._file_ouput_node.base_path = str(base_path)
        self._links.new(
            self._results_node.outputs[self._results_node_output_key],
            self._file_ouput_node.inputs["Image"],
        )
        self._suffix = suffix
        self._file_format = self._file_ouput_node.format.file_format.lower()

    def _place_node(
        self,
        node: bpy.types.Node,  # type: ignore
        prev_node: bpy.types.Node,  # type: ignore
        axis: bool,
    ) -> None:
        """
        Function that places a node near the previous one aligned along one axis.

        :param node: node to place, node
        :param prev_node: node to refer to, node
        :param axis: axis to align the node to, bool, 0 - vertical, 1 - horizontal
        """
        if axis == 1:
            offset = prev_node.width
        else:
            offset = prev_node.height
        node.location[abs(1 - axis)] = (
            prev_node.location[abs(1 - axis)] + offset + self._margin
        )
        node.location[axis] = prev_node.location[axis]

    @property
    def base_path(self) -> Path:
        return Path(self._file_ouput_node.base_path)

    @property
    def sub_path(self) -> Path:
        """
        `sub_path` is the remaining path between the general location where all images
        are saved to the final save location and the name of the image without file
        extension.
        """
        return Path(self._file_ouput_node.file_slots[0].path)

    @sub_path.setter
    def sub_path(self, sub_path: Path) -> None:
        final_sub_path = str(sub_path) + "_" + self._suffix
        self._file_ouput_node.file_slots[0].path = final_sub_path

    @property
    def full_path(self) -> Path:
        return Path(self.base_path, str(self.sub_path) + "0001." + self._file_format)

    @abstractmethod
    def _make_full_tree(self) -> tuple[CompositorNode, str]:  # type: ignore
        """
        The `_make_full_tree` method is responsible for creating the compositor tree
        as well as defining `self._result_node` as well as `self._output_field`.

        :returns: the node that contains the results to be exported as an image and the
        key to access the result node.
        """
        raise NotImplementedError(
            "Tree class requires a `_make_full_tree` method to construct the"
            + " compositor tree"
        )
