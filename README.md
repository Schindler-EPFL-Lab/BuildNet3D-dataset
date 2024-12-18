# BuildNet3D dataset

This repository contains the code for generating the BuildNet3D dataset.

## Installation

We use [UV](https://docs.astral.sh/uv/) to manage dependencies and the package.
To install the environment simply run `uv sync` in the base folder.

This should install the package in a python 3.10 virtual environment.

## Usage

To generate the dataset, run the following command:

```bash
"blender -b buildnet3d_dataset/blender_scripts/setup.blend --python buildnet3d_dataset/blender_scripts/dataset.py -- --output-dir ./outputs"
```

To change the type and number of building generated, change the config file in "buildnet3d_dataset/config/dataset_config.json".
The file should look something like:

```json
{
    "render_views": 1,  // number of view for each building
    "size": 1, // number of buildings generated
    "image_size": [
        500,
        500
    ],
    "roof_material": [
        "metall"
    ],
    "window_material": [
        "glass"
    ],
    "specific_views":[
        [90, 0, 0],
        [90, 0, 90],
        [90, 0, 180],
        [90, 0, 270]

    ]
}
```

To configure the segmentation use the "buildnet3d_dataset/config/segmentation_config.json" file.

## Contributing

We welcome contributions to the BuildNet3D dataset.
If you have any questions or suggestions, please open an issue or submit a pull request.

Code should be formatted using [Ruff](https://docs.astral.sh/ruff/).

## Citation

If you use the BuildNet3D dataset in your research, please cite the following paper:
