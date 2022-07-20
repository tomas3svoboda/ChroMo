import pandas as pd

"""
Object describing component in specific experiment
Contains:
    concentration_time (pandas.DataFrame) - describing progress of concentration in time
    feed_concentration (float) - concentration of component in feed
    name (string) - name of component
    experiment (Experiment) - reference to experiment
"""
class ExperimentComponent:
    concentration_time = pd.DataFrame()
    feed_concentration = 0.0
    name = ""
    experiment = 0