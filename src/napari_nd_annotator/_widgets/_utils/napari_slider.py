from superqt.sliders._labeled import QLabeledSlider
from superqt.sliders._sliders import QDoubleSlider
from qtpy.QtCore import Signal, QLocale


class QLabeledDoubleSlider(QLabeledSlider):
    _slider_class = QDoubleSlider
    _slider: QDoubleSlider
    _fvalueChanged = Signal(float)
    _fsliderMoved = Signal(float)
    _frangeChanged = Signal(float, float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setDecimals(2)
        locale = QLocale(QLocale.Language.C)
        self._label.setLocale(locale)

    def _setValue(self, value: float):
        """Convert the value from float to int before setting the slider value."""
        self._slider.setValue(value)

    def _rename_signals(self):
        self.valueChanged = self._fvalueChanged
        self.sliderMoved = self._fsliderMoved
        self.rangeChanged = self._frangeChanged

    def decimals(self) -> int:
        return self._label.decimals()

    def setDecimals(self, prec: int):
        self._label.setDecimals(prec)

