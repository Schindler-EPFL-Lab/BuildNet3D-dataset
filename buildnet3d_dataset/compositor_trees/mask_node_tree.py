from pathlib import Path

from bpy.types import (
    CompositorNode,  # type: ignore
    Node,  # type: ignore
)

from buildnet3d_dataset.compositor_trees.tree import Tree


class MaskNodeTree(Tree):
    def __init__(
        self,
        base_path: Path,
        nbr_modules: int,
        class_color_map: dict[int, tuple[float, float, float]],
    ) -> None:
        """
        `base_path` is the path into the dataset output directory. `nbr_modules` is the
        number of modules added to the volumes of the buildings.
        """
        self.nbr_modules = nbr_modules
        self.class_color_map = class_color_map
        super().__init__(base_path=Path(base_path, "masks"), suffix="mask")
        self._file_ouput_node.name = "Mask Output"

    def _make_full_tree(self) -> tuple[CompositorNode, str]:  # type: ignore
        """
        Function that creates a node tree with the necessary outputs to make
        segmentation masks.

        :return: the final node of the tree with the output image to be saved as well as
        the key to access final node output.
        """

        result_node = None
        for index in range(1, self.nbr_modules + 2):
            result_node = self._material_branch(index, result_node)

        return result_node, "Image"

    def _make_add_node(self, node1: Node, node2: Node) -> Node:  # type: ignore
        """
        Function that combines two nodes together summing their values.

        :param node1: first image node, node
        :param node2: second image node, node
        :return: resulting node, node
        """
        add_node = self._scene.node_tree.nodes.new(type="CompositorNodeMixRGB")
        add_node.blend_type = "Add".upper()
        _ = self._links.new(node1.outputs["Image"], add_node.inputs[1])
        _ = self._links.new(node2.outputs["Image"], add_node.inputs[2])
        return add_node

    def _make_color_node_rgb(self, red: float, green: float, blue: float) -> Node:  # type: ignore
        """
        Function that makes a mode with a color value (rgb).

        :return: color node
        """
        color_node = self._scene.node_tree.nodes.new(type="CompositorNodeCombineColor")
        color_node.mode = "RGB"
        color_node.inputs["Red"].default_value = red
        color_node.inputs["Green"].default_value = green
        color_node.inputs["Blue"].default_value = blue
        return color_node

    def _make_mask_id_node(self, index) -> Node:  # type: ignore
        """
        Function that takes input from the root node and renders one object id.

        :param index: index of the objects to render as a mask, int >= 0
        :return: mask_id_node, node
        """
        node = self._scene.node_tree.nodes.new(type="CompositorNodeIDMask")
        node.use_antialiasing = True
        node.index = index
        node.update()
        self._links.new(self._root_node.outputs["IndexOB"], node.inputs["ID value"])
        return node

    def _make_multiply_node(
        self,
        node1: Node,
        node2: Node,  # type: ignore
    ) -> Node:  # type: ignore
        """
        Function that combines two nodes together multiplying their values.

        :param node1: first image node, node
        :param node2: second image node, node
        :return: resulting node, node
        """
        multiply_node = self._scene.node_tree.nodes.new(type="CompositorNodeMixRGB")
        multiply_node.blend_type = "Multiply".upper()
        _ = self._links.new(node1.outputs["Alpha"], multiply_node.inputs[1])
        _ = self._links.new(node2.outputs["Image"], multiply_node.inputs[2])
        return multiply_node

    def _material_branch(self, index: int, result_node: None | Node = None) -> Node:  # type: ignore
        """
        Function that adds a mask material branch to the composite tree.
        :param index: index of the mask, int > 0
        :param result_node: previous resulting node to connect to the new branch, node
        :return: new resulting node, node
        """
        mask_id_node = self._make_mask_id_node(index)
        if result_node is not None:
            self._place_node(mask_id_node, result_node, True)
        else:
            self._place_node(mask_id_node, self._root_node, True)
        red = self.class_color_map[index][0]
        green = self.class_color_map[index][1]
        blue = self.class_color_map[index][2]
        color_node = self._make_color_node_rgb(red=red, green=green, blue=blue)
        self._place_node(color_node, mask_id_node, False)
        multiply_node = self._make_multiply_node(mask_id_node, color_node)
        self._place_node(multiply_node, mask_id_node, True)
        if result_node:
            add_node = self._make_add_node(result_node, multiply_node)
            self._place_node(add_node, multiply_node, True)
            return add_node
        return multiply_node
