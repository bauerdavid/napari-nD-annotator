from packaging import version
import napari


class StrComparableVersion(version.Version):
    def __init__(self, ver: version.Version):
        super().__init__(str(ver))

    def __eq__(self, other_ver):
        if type(other_ver) == str:
            other_ver = version.parse(other_ver)
        return super().__eq__(other_ver)

    def __gt__(self, other_ver):
        if type(other_ver) == str:
            other_ver = version.parse(other_ver)
        return super().__gt__(other_ver)

    def __ge__(self, other_ver):
        if type(other_ver) == str:
            other_ver = version.parse(other_ver)
        return super().__ge__(other_ver)

    def __lt__(self, other_ver):
        if type(other_ver) == str:
            other_ver = version.parse(other_ver)
        return super().__lt__(other_ver)

    def __le__(self, other_ver):
        if type(other_ver) == str:
            other_ver = version.parse(other_ver)
        return super().__le__(other_ver)


NAPARI_VERSION = StrComparableVersion(version.parse(napari.__version__))
__all__ = ["NAPARI_VERSION"]
