import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
interpolate_style_path = os.path.join(__location__, "interpolate_button.qss")
from napari.resources._icons import write_colorized_svgs, _theme_path
from napari.settings import get_settings

interpolate_icon_path = os.path.join(__location__, "interpolate.svg").replace("\\", "/")
settings = get_settings()
theme_name = settings.appearance.theme
out = _theme_path(theme_name)
write_colorized_svgs(
    out,
    svg_paths=[interpolate_icon_path],
    colors=[(theme_name, 'icon')],
    opacities=(0.5, 1),
    theme_override={'warning': 'warning', 'logo_silhouette': 'background'},
)
