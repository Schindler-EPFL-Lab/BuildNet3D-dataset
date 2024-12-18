# Synthetic Dataset Generator

This is a tool that generates a dataset of synthetic buildings of different typologies. 

[Orignial GitHub](https://github.com/CDInstitute/Building-Dataset-Generator) [Arxiv](https://arxiv.org/pdf/2104.12564v1.pdf) [Website](https://cdinstitute.github.io/Building-Dataset-Generator/) [Samples](https://drive.google.com/drive/folders/1_D9nuNd9VXjzdqMoKIqrET7yiq21uZv6?usp=sharing)

The generated data includes:

* Mesh files of generated buildings, ```.obj``` format
* Rendered images of the mesh, ```.png``` format
* Rendered segmentation masks, ```.png``` format
* Depth annotation, ```.png```
* Surface normals annotation, ```.png``` format
* Point cloud files, ```.ply``` format (the number of points by default is 2048, can be changed in ```dataset_config.py```)

## How To Use

* Install [Blender](https://www.blender.org/download/)>=3,6. After installation make sure to add blender as an Environment variable. 
* Download the package as a .zip file or:
```
git clone https://github.com/FlorentF9/semantic-sdf.git
```
* Navigate to the ```semantic-sdf/scripts/building_generator``` folder.
```
blender --python blender_library_install.py
```
This command line installs our package in blender's python. To create completely synthetic buildings, navigate to the '''semantic-sdf/building_generator''' folder and execute

```
blender setup.blend --python dataset.py
```

Unfortunately, it is not possible to use Blender in background mode as it will not render the image masks correctly.
Note:
all the parameters related to the dataset (including any specific parameters for your buildings (e.g. max and min height / width / length)) are to be provided in ```dataset_config.py```. Default values adhere to international standards (min) and most common European values (max):

* minimum height 3m
* minimum length and width 6m
* maximum length, width, height 30 m
Other values to set:
* number of dataset samples
* building types
* component materials
* rendered image dimensions
* number of points in the point clouds
* paths to store the generated data
* option to save the .exr files

### Annotation structure

{'img': 'images/0.png',
 'category': 'building',
'img_size': (256, 256),
'2d_keypoints': [],
'mask': 'masks/0.png',
 'img_source': 'synthetic',
 'model':  'models/0.obj',
 'point_cloud': 'PointCloud/0.ply',
 'model_source': 'synthetic',
 'trans_mat': 0,
 'focal_length': 35.0,
 'cam_position': (0.0, 0.0, 0.0),
 'inplane_rotation': 0,
 'truncated': False,
 'occluded': False,
 'slightly_occluded': False,
 'bbox': [0.0, 0.0, 0.0, 0.0],
 'material': ['concrete', 'brick']}

