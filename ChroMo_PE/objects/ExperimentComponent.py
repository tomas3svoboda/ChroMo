import pandas as pd


class ExperimentComponent:
    """Object describing component in specific experiment
    Contains:
    concentration_time (pandas.DataFrame) - describing progress of concentration in time
    feed_concentration (float) - concentration of component in feed
    name (string) - name of component
    experiment (Experiment) - reference to experiment
    """
    def __init__(self):
        self.concentrationTime = pd.DataFrame() # chromatogram
        self.feedConcentration = 0.0
        self.name = ""
        self.experiment = 0
        self.preprocessingScore = 1 # uncertainty score obtained from preprocessing
        self.inflectionWidth = 0 # width of the peak between rising and falling edge inflection points
        self.zeroSumOfResiduals = 0 # sum of error squares when the model prediction gives only zeros
