
"""
Object describing conditions of experiment
Contains:
    column_diameter (float) - diameter of column
    column_length (float) - length of column
    feed_volume (float) - volume of feed
    flow_rate (float) - flow rate
"""
class ExperimentCondition:
    def __init__(self):
        self.columnDiameter = -1.0
        self.columnLength = -1.0
        self.deadVolume = -1.0
        self._feedVolume = -1.0 # [mL]
        self._feedTime = -1.0 # [h]
        self._flowRate = -1.0 # [mL/h]
        self.originalFeedTime = -1.0

    @property
    def feedTime(self):
        return self._feedTime

    @property
    def feedVolume(self):
        return self._feedVolume

    @property
    def flowRate(self):
        return self._flowRate

    @feedTime.setter
    def feedTime(self, new_val):
        self._feedTime = new_val
        self._feedVolume = new_val * self.flowRate

    @feedVolume.setter
    def feedVolume(self, new_val):
        self._feedVolume = new_val
        self._feedTime = new_val/self.flowRate

    @flowRate.setter
    def flowRate(self, new_val):
        if self.flowRate != -1:
            raise Exception("Flow Rate can't be changed!")
        else:
            self._flowRate = new_val
