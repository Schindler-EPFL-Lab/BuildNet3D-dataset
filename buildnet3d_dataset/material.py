import bpy
import numpy as np


class Material:
    """
    Class that represents a material object in Blender.
    """

    def __init__(self, name: str, file_dir: str) -> None:
        self.name = (
            name.lower().capitalize()
        )  # name of the material and its texture folder
        self.filename = "material"
        self._path = file_dir + "/Textures/{}.blend".format(self.filename)
        self._add = "\\Material\\"
        self.value = self._load()  # loads material into the scene
        self.file_dir = file_dir
        self._update_nodes()  # loads the textures to the material

    def _load(self) -> bpy.types.Material:  # type: ignore
        try:
            return bpy.data.materials[self.name]  # type: ignore
        except KeyError:
            return self._load_new()

    def _load_maps(self, map_type: str) -> bpy.types.Image:  # type: ignore
        """
        Function that uploads texture map into the Material tree nodes. Textures
        are taken from folder Texture/Material where Material corresponds to the
        name of the material.

        :param map_type: type of the map to upload, str, one of 'Diffuse', 'Normal',
        'Roughness', 'Displacement'
        :param file_dir: absolut path to the Textures folder
        :return: texture map, bpy image object
        """
        assert map_type in ["Diffuse", "Normal", "Roughness", "Displacement"], (
            "Unknown map type, expected one of: 'Diffuse', 'Normal',"
            "'Roughness', 'Displacement'"
        )
        try:
            bpy.ops.image.open(  # type: ignore
                filepath=self.file_dir
                + "/Textures/{}/{}.png".format(self.name, map_type.capitalize())
            )
            _images = [x.filepath for x in bpy.data.images]  # type: ignore
            return bpy.data.images[  # type: ignore
                _images.index(
                    self.file_dir + "/Textures/{}/{}.png".format(self.name, map_type)
                )
            ]
        except Exception as e:
            print("Failed to load {} texture of {}".format(map_type, self.name))
            print(repr(e))

    def _load_new(self) -> bpy.types.Material:  # type: ignore
        try:
            # Load materials from scene material.blend to our scene
            with bpy.data.libraries.load(self._path, link=False) as (  # type: ignore
                data_from,
                data_to,
            ):
                data_to.materials = data_from.materials

            materials = [
                x
                for x in bpy.data.materials  # type: ignore
                if x.name.startswith("material")
            ]
            materials[-1].name = self.name
            return bpy.data.materials[self.name]  # type: ignore
        except Exception as e:
            print(repr(e))
            print("Could not import {} from {}".format(self.name, self._path))
            raise KeyboardInterrupt()

    def _update_nodes(self) -> None:
        """
        Function that updates the nodes of the material tree with the material
        textures.
        """
        if "Diffuse_texture" in [x.name for x in self.value.node_tree.nodes]:
            self.value.node_tree.nodes["Diffuse_texture"].image = self._load_maps(
                "Diffuse"
            )
            self.value.node_tree.nodes["Normal_texture"].image = self._load_maps(
                "Normal"
            )
            self.value.node_tree.nodes["Displacement_texture"].image = self._load_maps(
                "Displacement"
            )
        else:
            self.value = self._load_new()
            self._update_nodes()


class MaskMaterial(Material):
    def __init__(
        self,
        file_dir: str,
        name: str = "mask",
        filename: str = "mask",
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.filename = filename
        self.color = color
        Material.__init__(self, name, file_dir)

    def _update_nodes(self) -> None:
        self.value.node_tree.nodes["RGB"].outputs[0].default_value[0] = self.color[0]
        self.value.node_tree.nodes["RGB"].outputs[0].default_value[1] = self.color[1]
        self.value.node_tree.nodes["RGB"].outputs[0].default_value[2] = self.color[2]


class GlassMaterial(Material):
    def __init__(self, file_dir: str, name: str = "glass") -> None:
        Material.__init__(self, name, file_dir)

    def _load(self) -> bpy.types.Material:  # type: ignore
        try:
            return bpy.data.materials[self.name]  # type: ignore
        except Exception:
            bpy.ops.material.new()  # type: ignore
            _material = [
                x for x in bpy.data.materials if "Material" in x.name  # type: ignore
            ][-1]
            _material.name = self.name
            return _material

    def _update_nodes(self) -> None:
        if "Glass BSDF" not in [x.name for x in self.value.node_tree.nodes]:
            glass_node = self.value.node_tree.nodes.new("ShaderNodeBsdfGlass")
            glass_node.inputs["IOR"].default_value = 7.0
            glass_node.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)
            fresnel_node = self.value.node_tree.nodes.new("ShaderNodeFresnel")
            mix_node = self.value.node_tree.nodes.new("ShaderNodeMixShader")
            glossy_node = self.value.node_tree.nodes.new("ShaderNodeBsdfGlossy")
            glossy_node.inputs["Color"].default_value = (0.875, 0.875, 0.875, 1)
            output_node = self.value.node_tree.nodes["Material Output"]
            self.value.node_tree.links.new(mix_node.outputs[0], output_node.inputs[0])
            self.value.node_tree.links.new(fresnel_node.outputs[0], mix_node.inputs[0])
            self.value.node_tree.links.new(glass_node.outputs[0], mix_node.inputs[1])
            self.value.node_tree.links.new(glossy_node.outputs[0], mix_node.inputs[2])


class MetallMaterial(Material):
    def __init__(self, file_dir: str, name: str = "metall") -> None:
        Material.__init__(self, name, file_dir)

    def _load(self) -> bpy.types.Material:  # type: ignore
        try:
            return bpy.data.materials[self.name]  # type: ignore
        except Exception:
            bpy.ops.material.new()  # type: ignore
            _material = [
                x for x in bpy.data.materials if "Material" in x.name  # type: ignore
            ][-1]
            _material.name = self.name
            # Remove the default node so that the `_update_node` method runs
            principled_bsdf = bpy.data.materials["Metall"].node_tree.nodes[
                "Principled BSDF"
            ]
            bpy.data.materials["Metall"].node_tree.nodes.remove(principled_bsdf)
            return _material

    def _update_nodes(self) -> None:
        if "Principled BSDF" not in [x.name for x in self.value.node_tree.nodes]:
            self.value.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            node1 = self.value.node_tree.nodes["Principled BSDF"]
            _gray = np.random.uniform(0.05, 0.12)
            node1.inputs["Base Color"].default_value = [_gray, _gray, _gray, 1.0]
            node1.inputs["IOR"].default_value = 1.0

            node2 = self.value.node_tree.nodes["Material Output"]
            _ = self.value.node_tree.links.new(node1.outputs[0], node2.inputs[0])
