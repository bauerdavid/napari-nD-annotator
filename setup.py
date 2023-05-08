import sys
from setuptools import setup, Extension
import os
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

extra_compile_args = ["/std:c++17", "/openmp"] if sys.platform == "win32"\
    else ["-std=c++17"] if sys.platform == "darwin"\
    else ["-std=c++17", "-fopenmp"]
extra_link_args = [] if sys.platform in ["win32", "darwin"] else ["-lgomp"]
extensions = [
    Extension(
        "napari_nd_annotator.minimal_contour._eikonal_wrapper",
        ["src/napari_nd_annotator/minimal_contour/_eikonal_wrapper.pyx"],
        extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, language="c++", include_dirs=[np.get_include()]
    ),
    Extension(
        "napari_nd_annotator.mean_contour._essentials",
        ["src/napari_nd_annotator/mean_contour/cEssentialscy.pyx"],
        language="c++", include_dirs=[np.get_include()]
    ),
    Extension(
        "napari_nd_annotator.mean_contour._contour",
        ["src/napari_nd_annotator/mean_contour/contourcy.pyx"],
        language="c++", include_dirs=[np.get_include()]
    ),
    Extension(
        "napari_nd_annotator.mean_contour._reconstruction",
        ["src/napari_nd_annotator/mean_contour/reconstructioncy.pyx"],
        language="c++", include_dirs=[np.get_include()]
    )
]

CYTHONIZE = cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

setup(
    ext_modules=extensions,
    include_dirs=[np.get_include()]
)
