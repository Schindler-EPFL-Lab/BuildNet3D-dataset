import bpy

from buildnet3d_dataset.blender_utils import (
    deselect_all,
    get_min_max,
    select_all,
)


class Building:
    """
    Class that represents a building extruded from the contour, a single mesh.
    """

    def __init__(self, mesh):
        self.building = mesh

    def get_bb(self):
        """
        Function that gets the bounding box of the Building
        :return: bounding box, list of float
        [width_from, height_from, width_to, height_to]
        """
        _bb = list(get_min_max(self.building, 0)) + list(get_min_max(self.building, 1))
        return _bb

    def save(self, filename="test", ext="obj"):
        """
        Function that saves the building as a separate file.
        :param filename: name of the file to write without extension, str,
        default='test'
        :param ext: file extension, str, default='obj'
        :return:
        """
        deselect_all()
        self.building.select_set(True)
        bpy.ops.export_scene.obj(filepath=filename, use_selection=True)


class Iterator:
    """
    Class that iterates over a collection.
    """

    def __init__(self, collection, class_type):
        self.collection = collection
        self.index = 0
        self.class_type = class_type
        assert isinstance(self.collection, list), "Wrong iteration input."

    def __next__(self):
        try:
            _object = self.collection[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        assert isinstance(_object, self.class_type)
        return _object

    def __iter__(self):
        return self

    def has_next(self):
        return self.index < len(self.collection)


class BlenderReader:
    """
    Class that reads 3D file .gltf and manages the scene in blender.
    """

    def __init__(self, filename):
        self.filename = filename
        self._import()
        self.filename = self.filename.split(".")[-2]
        self.obj = bpy.data.objects
        self._clean()
        self.obj = bpy.data.objects

    def read(self):
        """
        Function that returns all the objects in the active scene.
        :return:
        """
        return self.obj

    def export(self, filename="test", ext="obj"):
        """
        Function that exports the whole scene as a given extension.
        :param filename: name of the file to write without extension, str,
        default='test'
        :param ext: name of the extension of the file to write, str,
        default='obj'
        :return:
        """
        select_all()
        if ext == "obj":
            bpy.ops.export_scene.obj(filepath="{}.{}".format(filename, ext))
        else:
            raise NotImplementedError
        print("File has been successfully saved as {}".format(filename))

    def _clean(self):
        """
        Function that cleans the scene from the excessive objects that do not
        belong to the model of interest.
        :return:
        """
        to_clean = [
            x
            for x in self.obj
            if x.parent and x.parent.name != self.filename.split(".")[0]
        ]
        deselect_all()
        for mesh in to_clean:
            try:
                mesh.select_set(True)
                bpy.ops.object.delete()
            except Exception:
                pass

    def _import(self):
        """
        Function that imports .gltf file into the scene.
        :return:
        """
        bpy.ops.import_scene.gltf(filepath=self.filename)
