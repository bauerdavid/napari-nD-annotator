# napari-nD-annotator
This repository contains some tools to make nD annotation easier.
The created modules are shown in the examples. First try `bounding_box_example.py`,
`hole_filling_example.py`, `projections_example.py` and `interpolation_example.py`, as `pipeline_examples.py`
uses all these modules.<br>
The main idea is to create bounding boxes around objects we want to annotate,
crop them, and annotate them one by one. This has mainly two advantages 
when visualizing in 3D:
1) We don't have to load the whole data into memory
2) The surrounding objects won't occlude the annotated ones, making it easier to check
the annotation.

## Installation
Clone the repository by
```
git clone https://github.com/bauerdavid/napari-nD-annotator/
```
It is advised to create a virtual environment
```
python -m virtualenv venv
source venv/bin/activate # activation on Linux
venv\Scripts\activate.bat # activation on Windows
```

You can install the required packages from the repository folder with
```
pip install -r requirements.txt
```

## Examples
For the examples first you should create `settings.py` in the root folder of the
repository (next to the examples). You can simply create a copy of 
`settings_template.py`, and then modify the parameters as needed.
### `bounding_box_example.py`
Here the BoundingBoxLayer is introduced. It can be used to draw nD bounding boxes,
and after that they can be moved, rescaled and deleted as needed. They are displayed
in 3D as well.
### `hole_filling_example.py`
A small widget is added to the window, which might help in the 
annotation process. One help is the `autofill objects` checkbox,
which will automatically close a started line drawn in a Labels layer and fill it.
Another tool is a converter between Labels and Shapes.
### `projections_example.py`
Here a widget is added to the window, which displays every projection of the data.
The projections will be automatically updated on hovering the mouse
over the image. Annotations in the Labels layer are also displayed.
If you click on a projection, the view will move to that projection.
### `interpolation_example.py`
This module can interpolate annotations between layers. Select the axis along you
want to generate missing layers. The currently displayed axes will be used as
spatial dimensions. Increasing the number of contour points will give nicer results,
but will cause a longer execution time.
### `pipeline_example.py`
The modules shown before combined in a pipeline for selecting single cells
for annotation. The pipeline goes as follows:
1) Create a `Points` layer
2) Put points in the middle of the cells you want to annotate
3) Click the `Create bounding boxes` button (module on the right)
4) Adjust the bounding boxes to match the cells
5) Click the `List objects` button
6) Select an object from the list
7) Annotate

Note: if you want to use the `autofill_objects` option, you must select
the "Cropped Mask" layer in the module.


