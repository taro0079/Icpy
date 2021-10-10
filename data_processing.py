from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class DataProcessing:
    def __init__(self, dataframe):
        self.current = dataframe['current']
        self.voltage = dataframe['voltage']
        max_current = np.max(self.current)
        self.xx = np.linspace(0, max_current, 1000)

    def interpolate(self):
        itp = InterpolatedUnivariateSpline(self.current, self.voltage, k=5)
        return itp(self.xx)

    def deleteBaseline(self):

