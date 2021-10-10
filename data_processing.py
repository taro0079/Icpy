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
        yy = self.interpolate()
        range_y = np.linspace(-10, 10, 1000)
        xx_size = len(self.xx)
        alldata = dict()
        sumdata = 1e10  # 絶対値の合計値
        for i in range_y:
            line = i * np.ones(xx_size)
            after_y = yy - line
            abs_after = np.abs(after_y)
            sum_after = np.sum(abs_after)
            if sum_after < sumdata:
                sumdata = sum_after
                alldata[sumdata] = i
        return yy - alldata[sumdata] * np.ones(xx_size)

    def calcIc(self):
        current_criterion = 1.0 # 電解基準
        fnew = InterpolatedUnivariateSpline(self.xx, self.deleteBaseline() - current_criterion)
        roots = fnew.roots()
        return roots[-1]
