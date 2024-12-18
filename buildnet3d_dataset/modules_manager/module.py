import io
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout

import bpy
import numpy as np

from buildnet3d_dataset.blender_utils import deselect_all, gancio, select
from buildnet3d_dataset.material_factory import MaterialFactory


class Module(ABC):
    def __init__(
        self,
        id: int,
        segmentation_color: tuple[float, float, float],
        materials: list[str],
        path_to_materials: str,
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        name: str = "generic",
        mesh: bpy.types.Mesh | None = None,  # type: ignore
    ) -> None:
        """
        Parent class for building any module.

        Any module has a `name`, a `dimensions` (`width`,`length`,`height`) and an `id`
        and `segmentation_color` that will be used for semantic segmentation. A list
        of materials for the module must be provided in `materials` as well as the
        absolute path to the texture folder in `path_to_materials`. A module must have
        a `mesh`; if none is given, a mesh is generated using the module's `name`.
        Finally, the module can be associated to a `volume` to which it will be
        attached.
        """
        self.id = id
        self.segmentation_color = (
            segmentation_color[0],
            segmentation_color[1],
            segmentation_color[2],
            1,
        )
        self.name = name
        self.connector = None
        self.width = width
        self.length = length
        self.height = height
        self.dimensions = (width, length, height)
        self.mesh = mesh
        if self.mesh is None:
            self.mesh = self._create()
        try:
            self.parent = self._nest()
        except Exception:
            pass
        self.y_offset = 0
        self.material = materials
        if self.mesh is not None:
            self.mesh["inst_id"] = id
            self.mesh.pass_index = id
            self.dimensions = (
                self.mesh.dimensions[0],
                self.mesh.dimensions[1],
                self.mesh.dimensions[2],
            )
        self.path_to_materials = path_to_materials

    def __copy__(self) -> "Module":
        if self.name is None:
            raise RuntimeError("Need to provide a name to the module")
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        deselect_all()
        select(self.mesh)
        bpy.ops.object.duplicate_move()  # type: ignore
        _name = self.mesh.name.split(".")[0]

        ind = [
            x.name
            for x in bpy.data.objects  # type: ignore
            if _name + "." in x.name or _name == x.name
        ]
        mesh = bpy.data.objects[ind[-1]]  # type: ignore
        m = self.__class__(
            self.id,
            self.segmentation_color[:3],
            self.material,
            self.path_to_materials,
            self.width,
            self.length,
            self.height,
            self.name,
            mesh=mesh,
        )
        # self._triangulate()

        if self.connector:
            m.connect(self.connector.axis, self.connector.side)
        return m

    def apply(self) -> None:
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        if len(self.material) > 0:
            _material = np.random.choice(self.material)
            _material = MaterialFactory(self.path_to_materials).produce(_material)
        else:
            _material = MaterialFactory(self.path_to_materials).produce()
        self.mesh.active_material = _material.value

    def position(self, position: list | tuple | np.ndarray) -> None:
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        assert len(position) == 3, "Position should have 3 values, " "got {}".format(
            len(position)
        )

        for i in range(len(position)):
            self.mesh.location[i] += position[i]

    def remove(self) -> None:
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        deselect_all()
        bpy.data.objects[self.mesh.name].select_set(True)  # type: ignore
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stdout):
            bpy.ops.object.delete()  # type: ignore

    def connect(self, axis: int, side: int = 0) -> None:
        self.connector = self.ModuleConnector(self, axis, side)

    @abstractmethod
    def _create(self) -> None:
        # rule how connects to mesh
        raise NotImplementedError

    def _nest(self) -> bpy.types.Collection:  # type: ignore
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        deselect_all()
        postfix = "_0"
        _name = self.name + postfix
        if _name not in [
            x.name
            for x in bpy.data.collections[self.volume.name].children  # type: ignore
        ]:
            bpy.data.collections.new(_name)  # type: ignore
            _name = [
                x.name
                for x in bpy.data.collections
                if _name in x.name  # type: ignore
            ][-1]
            bpy.data.collections[self.volume.name].children.link(  # type: ignore
                bpy.data.collections[_name]  # type: ignore
            )

        bpy.data.collections[_name].objects.link(  # type: ignore
            bpy.data.objects[self.mesh.name]  # type: ignore
        )
        return bpy.data.collections[_name]  # type: ignore

    def _old_nest(self) -> bpy.types.Collection:  # type: ignore
        if self.mesh is None:
            raise RuntimeError("Need to provide a mesh to the module")
        deselect_all()
        names = [x.name for x in bpy.data.collections]  # type: ignore
        if self.name not in names:
            bpy.data.collections.new(self.name)  # type: ignore
            bpy.data.collections["Building"].children.link(  # type: ignore
                bpy.data.collections[self.name]  # type: ignore
            )
        bpy.data.collections[self.name].objects.link(  # type: ignore
            bpy.data.objects[self.mesh.name]  # type: ignore
        )
        return bpy.data.collections[self.name]  # type: ignore

    def _rename_material(self) -> None:
        ind = bpy.data.objects[self.name].active_material_index  # type: ignore
        bpy.data.materials[ind].name = "module_{}".format(self.name)  # type: ignore

    def _remove_material(self) -> None:
        """
        Function that removes a material from the scene.
        :return:
        """
        if self.mesh is not None and self.mesh.active_material:
            bpy.data.materials.remove(  # type: ignore
                self.mesh.active_material, do_unlink=True
            )

    def _triangulate(self) -> None:
        deselect_all()
        if self.mesh:
            select(self.mesh)
            bpy.ops.object.modifier_add(type="TRIANGULATE")  # type: ignore
            bpy.ops.object.modifier_apply()  # type: ignore

    @property
    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError("Abstract method call")

    class ModuleConnector:
        """This is conecter for a standard module"""

        def __init__(self, module: "Module", axis: int, side: int = 0) -> None:
            self.module = module
            self.axis = axis
            self.side = side
            self._connect()

        def _connect(self) -> None:
            gancio(
                self.module.volume,
                self.module,
                axis=self.axis,
                border1=self.side,
                border2=0,
            )
