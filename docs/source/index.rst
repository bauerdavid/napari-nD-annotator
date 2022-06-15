.. napari-nD-annotator documentation master file, created by
   sphinx-quickstart on Wed Jun 15 09:55:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to napari-nD-annotator's documentation!
===============================================

A toolbox for annotating objects one by one in nD

This plugin contains some tools to make 2D/3D, but basically any dimensional
annotation easier. Main features:

* nD bounding box layer
* object list from bounding boxes
* visualizing selected objects from different projections
* auto-filling labels
* label slice interpolation

The main idea is to create bounding boxes around objects we want to annotate,
crop them, and annotate them one by one. This has mainly two advantages when
visualizing in 3D:

We don't have to load the whole data into memory
The surrounding objects won't occlude the annotated ones, making it easier to
check the annotation.

.. note::
   This documentation is currently in progress.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   Usage

