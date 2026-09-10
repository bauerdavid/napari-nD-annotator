import os

from .interpolation import interpolate_style_path
from .mc_contour import mc_contour_style_path

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
loading_icon_path = os.path.join(__location__, "loading.svg")
