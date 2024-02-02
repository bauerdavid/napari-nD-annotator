# napari-nD-annotator

[![License BSD-3](https://img.shields.io/pypi/l/napari-nD-annotator.svg?color=green)](https://github.com/bauerdavid/napari-nD-annotator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nD-annotator.svg?color=green)](https://pypi.org/project/napari-nD-annotator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nD-annotator.svg?color=green)](https://python.org)
[![tests](https://github.com/bauerdavid/napari-nD-annotator/workflows/tests/badge.svg)](https://github.com/bauerdavid/napari-nD-annotator/actions)
[![codecov](https://codecov.io/gh/bauerdavid/napari-nD-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/bauerdavid/napari-nD-annotator)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-nD-annotator)](https://napari-hub.org/plugins/napari-nD-annotator)

**A toolbox for annotating objects one by one in nD.**

This plugin contains some tools to make 2D/3D (and technically any dimensional) annotation easier.
Main features:
 * auto-filling labels
 * label slice interpolation (geometric mean, RPSV representation)
 * minimal contour segmentation

If the <code>[napari-bbox]</code> plugin is also installed (see [Installation](#installation)), you can also
 * list objects annotated with bounding boxes 
 * visualize selected objects from different projections

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

The plugin is also available in [napari-hub], to install it directly from napari, please refer to
[plugin installation instructions] at the official [napari] website.


### Optional packages
There are some functionalities which require additional Python packages.

#### Bounding boxes
The bounding box and object list functionality requires the <code>[napari-bbox]</code> Python package.
If you want to use these features, install <code>[napari-bbox]</code> separately either using [pip] or directly from napari.
You can also install it together with this plugin:
```
pip install napari-nD-annotator[bbox]
```
> [!WARNING]
> The <code>[napari-bbox]</code> plugin currently works only with `napari<=0.4.17`. Do not install it with newer versions.

#### Minimal surface
To use the minimal surface method, you will need the <code>[minimal-surface]</code> Python package as well. Please install it using [pip]:

Separately:
```
pip install minimal-surface
```

Or bundled with the plugin:
```
pip install napari-nD-annotator[ms]
```
> [!WARNING]
> The <code>[minimal-surface]</code> package is only available for Windows at the time. We are actively working on bringing it to Linux and Mac systems as well.

#

If you would like to install all optional packages, use
```
pip install napari-nD-annotator[all]
```
###
If any problems occur during installation or while using the plugin, please [file an issue].

## Usage
You can start napari with the plugin's widgets already opened as:

    napari -w napari-nD-annotator "Object List" "Annotation Toolbox"


### Bounding boxes
The main idea is to create bounding boxes around objects we want to annotate, crop them, and annotate them one by one. This has mainly two advantages when visualizing in 3D:

1. We don't have to load the whole data into memory
2. The surrounding objects won't occlude the annotated ones, making it easier to check the annotation.

Bounding boxes can be created from the `Object list` widget. The dimensionality of the bounding box layer will be determined from the image layer. As bounding boxes are created, a small thumbnail will be displayed.

The proposed pipeline goes as follows:

 1. Create a bounding box layer
 2. Select data parts using the bounding boxes
 3. Select an object from the object list
 4. Annotate the object
 5. Repeat from 3.

### Slice interpolation
The `Interpolation` tab contains tools for estimating missing annotation slices from existing ones. There are multiple options:
 * Geometric: the interpolation will be determined by calculating the average of the corresponding contour points.
 * RPSV: A more sophisticated average contour calculation, see the preprint [here](https://arxiv.org/pdf/1901.02823.pdf).
 * Distance-based: a signed distance transform is applied to the annotations. The missing slices will be filled in using their 
weighted sum.

> **Note**: Geometric and RPSV interpolation works only when there's a single connected mask on each slice. If you want to 
> interpolate disjoint objects (*e.g.* dividing cells), use distance based interpolation instead.

> **Note**: Distance-based interpolation might give bad results if some masks are too far away from each other on the same slice
> and there's a big offset compared to the other slice used in the interpolation. If you get unsatisfactory results, try
> annotating more slices (skip less frames).

https://user-images.githubusercontent.com/36735863/188876826-1771acee-93ba-4905-982e-bfb459329659.mp4

### Minimal contour
This plugin can estimate a minimal contour, which is calculated from a point set on the edges of the object, which are provided by the user. This contour will follow some kind of image feature (pixels with high gradient or high/low intensity).
Features:
 * With a single click a new point can be added to the set. This will also extend the contour with the curve shown in red
 * A double click will close the curve by adding both the red and gray curves to the minimal contour
 * When holding `Shift`, the gray and red highlight will be swapped, so the other curve can be added to the contour
 * With the `Ctrl` button down a straight line can be added instead of the minimal path
 * If the anchor points were misplaced, the last point can be removed by right-clicking, or the whole point set can be cleared by pressing `Esc`
 * The `Param` value at the widget will decide, how strongly should the contour follow edges on the image. Higher value means higher sensitivity to image data, while a lower value will be closer to straight lines.
 * Different features can be used, like image gradient or pixel intensities, and also user-defined features (using Python)
   * the image is accessed as the `image` variable, and the features should be stored in the `features` variable in the small code editor widget

This functionality can be used by selecting the `Minimal Contour` tab in the `Annotation Toolbox` widget, which will create a new layer called `Anchors`.

> **Warning**: Do not remove the `Anchors` layer!

> **Warning**: Some utility layers appear in the layer list when using the plugin. These are marked with a lock (:lock:) symbol.
> __Do not remove them or modify their data, as this will most probably break the plugin!__ However, you can change their appearance,
> *e.g.* their color settings.

#### Intensity-based:

https://user-images.githubusercontent.com/36735863/191023482-0dfafb5c-003a-47f6-a21b-8582a4e3930f.mp4

#### Gradient-based:

https://user-images.githubusercontent.com/36735863/191024941-f20f63a0-8281-47d2-be22-d1ec34fe1f5d.mp4

#### Custom feature:

https://user-images.githubusercontent.com/36735863/191025028-3f807bd2-1f2e-40d2-800b-48af820a7dbe.mp4

### Shortcuts

| Action                                        | Mouse               | Keyboard       |
|-----------------------------------------------|---------------------|----------------|
| Increment selected label                      | `Shift + Wheel ⬆️`  | `E`            |
| Decrement selected label                      | `Shift + Wheel ⬇️`  | `Q`            |
| Previous slice                                | `Ctrl + Wheel ⬆️`\* | `A`            |
| Next slice                                    | `Ctrl + Wheel ⬇️`\* | `D`            |
| Increase paint brush size of labels layer     | `Alt + Wheel ⬆️`    | `W`            |
| Decrease paint brush size of labels layer     | `Alt + Wheel ⬇️`    | `S`            |
| Interpolate                                   | -                   | `Ctrl+I`       |
| Change between 'Anchors' and the labels layer | -                   | `Ctrl+Tab`     |
| Jump to layer `#i`                            | -                   | `Ctrl+'i'`\*\* |

> *Built-in functionality of [napari]
> 
> **`i`: 0-9

> **Note**: you can check the list of available shortcuts by clicking the `?` button in the bottom right corner of the main widget.

## License

Distributed under the terms of the [BSD-3] license,
"napari-nD-annotator" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[napari-hub]: https://napari-hub.org/
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[plugin installation instructions]: https://napari.org/plugins/find_and_install_plugin.html
[file an issue]: https://github.com/bauerdavid/napari-nD-annotator/issues/new/choose
[napari-bbox]: https://github.com/bauerdavid/napari-bbox
[minimal-surface]: https://pypi.org/project/minimal-surface
