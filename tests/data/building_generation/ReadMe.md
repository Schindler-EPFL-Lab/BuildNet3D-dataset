# ReadMe Building Generation Mock Files

## General

`mock_segmentation_mapping.json` is a segmentation mapping file for the mock models below. 

`vertex_test.csv` contains verticies with test colours to be fixed

`vertex_truth.csv` contains the data from `vertex_test.csv` corrected and segmented with regards to `mock_segmentation_mapping.json`

## Cubes Model

`mock_model_cubes.ply` is a 3D point cloud model with exported meshes. It contains coordinates, normals, colour, s, and t values. The 3D model is setup according to `mock_model_cubes.blend`. It has 8 1m x 1m x 1m cubes each one assigned to a different class. The classes are assigned as colour attributes according to the color mapping in `mock_segmentation_mapping.json`.

Cube locations (x, y, z):
- other_class cube is located at 2m, -2m, -2m.
- other_class cube 2 is located at 5m, -2m, -2m.
- wall cube is located at -2m, 2m, -2m.
- wall cube 2 is located at -2m, 5m, -2m.
- window cube is located at -2m, -2m, 2m.
- window cube 2 is located at -2m, -2m, 2m.
- roof cube is located at 4m, -2m, 4m.
- roof cube 2 is located at 4m, -2m, 6m.

`mock_model_cubes.blend` is the blender file used to create `mock_model_cubes.ply`.

`mock_model_cubes_mesh_info_truth.csv` is a csv file containing the fixed and segmented mesh information for `mock_model_cubes.ply` that can be imported as a `DataFrame` to test functions that read the mesh info in `mock_model_cubes.ply`.

`mock_model_cubes_mesh_normal_info_truth.csv` is a csv file containing the correct nroaml information for the meshes in `mock_model_cubes_mesh_info_truth.csv`. This can be imported as a DataFrame to test functions that read the mesh info in `mock_model_cubes.ply`.

## House Model

`mock_model_house.ply` is a 3D point cloud model with exported meshes. It contains coordinates, normals, colour, s, and t values. The 3D model is setup according to `mock_model_house.blend`. It has a small model house positioned so that the base of the house is resting cenetered at (0m, 0m). The body of the house consists of a 10m x 20m x 4m (width x length x height) rectangular prism with 1 window on each side and a roof. The windows are placed ontop of the building surface such that they do not protrude through the surface of the building. The roof is place ontop of the building such that it doesnot protrude through the top of the building. The windows on the sides of the house measuring 20m x 4m are 0.2m x 16m x 2m and the windows on the sides of the house measuring 10m x 4m are 6m x 0.2m x 2m. The roof is a large retangular prism that is cenetered on the building and is 12m x 22m x 1m.

`mock_model_house.blend` is the blender file used to create `mock_model_house.ply`