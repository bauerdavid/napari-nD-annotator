import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
mc_contour_style_path = os.path.join(__location__, "mc_contour_button.qss")
from napari.resources._icons import write_colorized_svgs, _theme_path
from napari.settings import get_settings

mc_contour_icon_path = os.path.join(__location__, "mc_contour.svg").replace("\\", "/")
settings = get_settings()
theme_name = settings.appearance.theme
out = _theme_path(theme_name)
write_colorized_svgs(
    out,
    svg_paths=[mc_contour_icon_path],
    colors=[(theme_name, 'icon')],
    opacities=(0.5, 1),
    theme_override={'warning': 'warning', 'logo_silhouette': 'background'},
)
