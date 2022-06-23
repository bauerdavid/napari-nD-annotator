# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information -----------------------------------------------------

project = 'napari-nD-annotator'
copyright = '2022, David Bauer'
author = 'David Bauer'

# The full version, including alpha/beta/rc tags
import napari_nd_annotator
release = napari_nd_annotator.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autosummary_context = \
    {
        "ignored_members": {
            "BoundingBoxLayer": [
                "mouse_move_callbacks",
                "mouse_wheel_callbacks",
                "mouse_drag_callbacks",
                "mouse_double_click_callbacks"
            ],
            "boundingbox": [
                "napari_nd_annotator.boundingbox.bounding_box",
                "napari_nd_annotator.boundingbox.bounding_boxes"
            ],
            "_widgets": [
                "napari_nd_annotator._widgets.annotator_module",
                "napari_nd_annotator._widgets.interpolation_widget",
                "napari_nd_annotator._widgets.object_list",
                "napari_nd_annotator._widgets.projections"
            ]
        }
    }

add_module_names = False
napoleon_google_docstrings = False
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']