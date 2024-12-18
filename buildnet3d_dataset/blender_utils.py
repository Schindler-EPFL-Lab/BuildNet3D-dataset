import logging

import bpy
import numpy as np
from bpy.types import Object  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

from buildnet3d_dataset.iou import Intersection

import bmesh  # isort: skip # type: ignore


def deselect_all() -> None:
    """
    Function that deselects all the objects in the scene.
    :return: None
    """
    for obj in bpy.data.objects:  # type: ignore
        obj.select_set(False)


def select_all() -> None:
    """
    Function that selects all objects in the scene.
    """
    for obj in bpy.data.objects:  # type: ignore
        obj.select_set(True)


def extrude(mesh, height, direction=-1):
    """
    Function that extrudes a given plane to a given height in a given direciton.
    :param mesh: plane to extrude, Blender plane object mesh
    :param height: height to extrude the plane to, float or int
    :param direction: direction to make the extrusion into,
                -1 -> top
                1 -> bottom,
                default = -1
    :return:
    """
    assert issubclass(height.__class__, int) or issubclass(height.__class__, float), (
        "Expected height as a float or " "an int, got {}".format(type(height))
    )
    assert direction in [-1, 1], "Expected direction to be -1 or 1, got {}".format(
        direction
    )

    mesh.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")  # Change to face selection
    bpy.ops.mesh.select_all(action="SELECT")  # Select all faces

    bm = bmesh.new()
    bm = bmesh.from_edit_mesh(bpy.context.object.data)

    # Extude Bmesh
    for f in bm.faces:
        face = f.normal
    r = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    verts = [e for e in r["geom"] if isinstance(e, bmesh.types.BMVert)]
    TranslateDirection = face * direction * height  # Extrude Strength/Length
    bmesh.ops.translate(bm, vec=TranslateDirection, verts=verts)

    # Update & Destroy Bmesh
    bmesh.update_edit_mesh(bpy.context.object.data)  # Write the bmesh back to the mesh
    bm.free()  # free and prevent further access

    # Flip normals
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()

    # At end recalculate UV
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()

    # Switch back to Object at end
    bpy.ops.object.mode_set(mode="OBJECT")

    # Origin to center
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")


def get_min_max(volume, axis):
    """
    Function that returns limits of a mesh on the indicated axis. Only applied
    to objects that are not rotated or rotated to 90 degrees
    :param volume: volume to get the dims of, mesh
    :param axis: int, 0 - width; 1 - length; 2 - height
    :return: min, max, float
    """
    bpy.context.view_layer.update()
    bb_vertices = [Vector(v) for v in volume.bound_box]
    mat = volume.matrix_world
    world_bb_vertices = [mat @ v for v in bb_vertices]
    return min([x[axis : axis + 1][0] for x in world_bb_vertices]), max(
        [x[axis : axis + 1][0] for x in world_bb_vertices]
    )


def gancio(
    v1: "Volume",  # noqa: F821
    v2: "Volume",  # noqa: F821
    axis: int,
    border1: int = 0,
    border2: int = 0,
    overlap: float = 0,
) -> None:
    """
    Function that attaches `v2` to `v1` on either the x or y axis based on conditions
    set by `axis`, `border1`, and `border2`, with an `overlap`.

    `axis=0` corresponds to the x-axis and `axis=1` corresponds to the y axis.
    `border1` describes the min (`border1=0`) or max side (`border1=1`) of `axis`
    `border2` descirbes the min (`border2=0`) or max side (`border1=1`) of the
    opposite axis.
    """
    mapping = {0: -1, 1: 1}
    coords1 = [get_min_max(v1.mesh, 0), get_min_max(v1.mesh, 1)]
    coords2 = [get_min_max(v2.mesh, 0), get_min_max(v2.mesh, 1)]

    v2.mesh.location[axis] = coords1[axis][border1] + (
        0.5 * (coords2[axis][1] - coords2[axis][0]) * mapping[border1]
        - overlap * mapping[border1]
    )

    v2.mesh.location[abs(1 - axis)] = (
        coords1[abs(1 - axis)][abs(1 - border2)]
        + mapping[border2] * (coords1[abs(1 - axis)][1] - coords1[abs(1 - axis)][0])
        + (
            0.5
            * (coords2[abs(1 - axis)][1] - coords2[abs(1 - axis)][0])
            * mapping[abs(1 - border2)]
        )
    )


def gancio2(v1, v2, axis, border1=0, border2=0):
    """
    Function that attaches one volume to another one based on condition.
    :param v1: volume to attach the other volume to, Volume or Module
    :param v2: volume to attach to the other volume, Volume or Module
    :param axis: axis along which the volume will be attached, bool,
                            0 - x axis,
                            1 - y axis
    :param border1: max or min side of the axis, 0 - min, 1 - max
    :param border2: max or min side of the opposite axis, 0 - min, 1 - max
    :return:
    """

    v2.mesh.rotation_euler[2] = 0
    v2.mesh.location[2] = 0
    place(v1, v2, axis, border1, border2)
    _intersections = []

    iou = Intersection(v1, v2)

    for i in range(8):
        deselect_all()
        v2.mesh.rotation_euler[2] += np.radians(90)
        place(v1, v2, axis, border1, border2)

        if border2 == 1:
            v2.mesh.location[abs(1 - axis)] += 0.5
        else:
            v2.mesh.location[abs(1 - axis)] -= 0.5

        inter = iou.calculate()
        if i < 4:
            _intersections.append(inter)
        if i >= 4:
            if inter == min(_intersections):
                v2.mesh.rotation_euler[2] -= np.radians(360)
                break

    place(v1, v2, axis, border1, border2)


def gancio3(v1, v2, axis, border1=0, border2=0):
    """
    Function that attaches one volume to another one based on condition.
    :param v1: volume to attach the other volume to, Volume or Module
    :param v2: volume to attach to the other volume, Volume or Module
    :param axis: axis along which the volume will be attached, bool,
                                0 - x axis,
                                1 - y axis
    :param border1: max or min side of the axis, 0 - min, 1 - max
    :param border2: max or min side of the opposite axis, 0 - min, 1 - max
    :return:
    """

    v2.mesh.rotation_euler[2] = 0
    place(v1, v2, axis, border1, border2)
    _intersections = []

    iou = Intersection(v1, v2)

    for i in range(8):
        deselect_all()
        v2.mesh.rotation_euler[2] += np.radians(90)
        place(v1, v2, axis, border1, border2)

        if border2 == 1:
            v2.mesh.location[abs(1 - axis)] += 0.5
        else:
            v2.mesh.location[abs(1 - axis)] -= 0.5

        inter = iou.calculate(i)
        if i < 4:
            _intersections.append(inter)
        if i >= 4:
            if inter == min(_intersections):
                v2.mesh.rotation_euler[2] -= np.radians(360)
                break

    place(v1, v2, axis, border1, border2)
    if border2 == 1:
        v2.mesh.location[abs(1 - axis)] += 0.5
    else:
        v2.mesh.location[abs(1 - axis)] -= 0.5


def place(v1, v2, axis, border1, border2):
    mapping = {0: -1, 1: 1}
    coords1 = [get_min_max(v1.mesh, 0), get_min_max(v1.mesh, 1)]  # volume min max
    v2.mesh.location[axis] = coords1[axis][border1]  # border1 - front or back
    v2.mesh.location[abs(1 - axis)] = coords1[abs(1 - axis)][border2] + mapping[
        abs(1 - border2)
    ] * np.diff(coords1[abs(1 - axis)])  # border2 start or end


def intersection_check(v1: Object, v2: Object) -> bool:
    bm1 = bmesh.new()
    bm2 = bmesh.new()

    # fill bmesh data from objects
    bm1.from_mesh(v1.data)
    bm2.from_mesh(v2.data)

    # fixed it here:
    bm1.transform(v1.matrix_basis)
    bm2.transform(v2.matrix_basis)

    # make BVH tree from BMesh of objects
    v1_BVHtree = BVHTree.FromBMesh(bm1)
    v2_BVHtree = BVHTree.FromBMesh(bm2)

    # get intersecting pairs
    inter = v1_BVHtree.overlap(v2_BVHtree)
    if len(inter) > 0:
        return True
    return False


def select(_volume):
    _volume.select_set(True)
    bpy.context.view_layer.objects.active = _volume


def top_connect(volume, module):
    """
    Function that connects a module to the top of the volume (roof)
    :param volume: volume to connect the module to
    :param module: module to connect to the volume
    :return:
    """
    volume_top = get_min_max(volume.mesh, 2)[1]
    coords = get_min_max(module.mesh, 2)

    module.mesh.location[2] = volume_top + ((coords[1] - coords[0]) / 2)
    for axis in range(2):
        module.mesh.location[axis] = volume.mesh.location[axis]
    # check
    coords = get_min_max(module.mesh, 2)
    if coords[0] > volume_top:
        module.mesh.location[2] -= coords[0] - volume_top


def mesh_boolean_union(base_mesh: Object, union_mesh: Object) -> bool:
    """
    Joins the `union_mesh` onto the `base_mesh` if they are touching, taking all
    attributes from the `base_mesh`.

    :returns: True of operation was completed and False if it was not.
    """
    deselect_all()

    if not intersection_check(base_mesh, union_mesh):
        logging.debug(
            f"Mesh Boolean union between {base_mesh.name} and {union_mesh.name} not "
            + "possible as meshes are not overlapped."
        )
        return False
    bpy.context.view_layer.objects.active = bpy.data.objects[base_mesh.name]
    bpy.ops.object.modifier_add(type="BOOLEAN")
    bpy.data.objects[base_mesh.name].modifiers["Boolean"].operation = "UNION"
    bpy.data.objects[base_mesh.name].modifiers["Boolean"].object = bpy.data.objects[
        union_mesh.name
    ]
    bpy.data.objects[base_mesh.name].modifiers["Boolean"].solver = "EXACT"
    bpy.ops.object.modifier_move_to_index(modifier="Boolean", index=0)
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects[union_mesh.name].select_set(True)
    bpy.ops.object.delete()
    return True


def dissolve_faces(obj: Object) -> None:
    """
    Dissolves all excess faces from `mesh` so that all is left is the net of the object.
    """
    deselect_all()
    if obj.type != "MESH":
        raise TypeError("Object provided is not 'MESH' type")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_limited(use_dissolve_boundaries=True)
    bpy.ops.object.mode_set(mode="OBJECT")
