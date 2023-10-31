from superqt.sliders import QDoubleRangeSlider


class QSymmetricDoubleRangeSlider(QDoubleRangeSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valueChanged.connect(self.change_symmetrically)
        self._prev_value = self.value()

    def change_symmetrically(self, value):
        try:
            if value[0] != self._prev_value[0] and value[1] != self._prev_value[1]:
                return
            if value[0] != self._prev_value[0]:
                diff = value[0] - self._prev_value[0]
                value = (value[0], value[1]-diff)
            elif value[1] != self._prev_value[1]:
                diff = value[1] - self._prev_value[1]
                value = (value[0]-diff, value[1])
            self.valueChanged.disconnect(self.change_symmetrically)
            self.setValue(value)
            self.valueChanged.connect(self.change_symmetrically)
        finally:
            self._prev_value = value
