
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
        self.columnDiameter = 0.0
        self.olumnLength = 0.0
        self.feedVolume = 0.0
        self.flowRate = 0.0