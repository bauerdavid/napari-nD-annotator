# napari-nD-annotator

[![License BSD-3](https://img.shields.io/pypi/l/napari-nD-annotator.svg?color=green)](https://github.com/bauerdavid/napari-nD-annotator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nD-annotator.svg?color=green)](https://pypi.org/project/napari-nD-annotator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nD-annotator.svg?color=green)](https://python.org)
[![tests](https://github.com/bauerdavid/napari-nD-annotator/workflows/tests/badge.svg)](https://github.com/bauerdavid/napari-nD-annotator/actions)
[![codecov](https://codecov.io/gh/bauerdavid/napari-nD-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/bauerdavid/napari-nD-annotator)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-nD-annotator)](https://napari-hub.org/plugins/napari-nD-annotator)

A toolbox for annotating objects one by one in nD

This plugin contains some tools to make 2D/3D, but basically any dimensional annotation easier.
Main features:
 * nD bounding box layer
 * object list from bounding boxes
 * visualizing selected objects from different projections
 * auto-filling labels
 * label slice interpolation

The main idea is to create bounding boxes around objects we want to annotate, crop them, and annotate them one by one. This has mainly two advantages when visualizing in 3D:

1. We don't have to load the whole data into memory
2. The surrounding objects won't occlude the annotated ones, making it easier to check the annotation.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

You can install `napari-nD-annotator` via [pip]:

    pip install napari-nD-annotator

The plugin is also available in napari-hub, to install it directly from napari, please refer to
[plugin installation instructions] at the official [napari] website.

## Usage
You can start napari with the plugin's widgets already opened as:

    napari -w napari-nD-annotator "Object List" "Annotation Toolbox"

The proposed pipeline goes as follows:

 1. Create a bounding box layer
 2. Select data parts using the bounding boxes
 3. Select an object from the object list
 4. Annotate the object
 5. Repeat from 3.

## Example

    import napari
    from skimage.data import cells3d
    import numpy as np
    viewer = napari.Viewer()
    nuclei = cells3d()[:, 1]
    viewer.add_image(nuclei, colormap="magma")
    viewer.add_labels(np.zeros_like(nuclei))
    napari.run()

![](https://i.imgur.com/xZxdvEQ.gif)

## License

Distributed under the terms of the [BSD-3] license,
"napari-nD-annotator" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[plugin installation instructions]: https://napari.org/plugins/find_and_install_plugin.html
[file an issue]: https://github.com/bauerdavid/napari-nD-annotator/issues/new/choose
