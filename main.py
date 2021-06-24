import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate


class dataRepo():

    # PATHからデータを読み込み
    def readData(self, path):
        f = open(path)
        lines = f.readlines()[1:-1]
        splitlines = [i.split() for i in lines]
        floatlines = [map(float, i) for i in splitlines]
        df = pd.DataFrame(floatlines)
        df.columns = ["sec(s)", "current(A)", "voltage(V)",
                      "temp1", "temp2", "temp3"]
        df = df.drop_duplicates(subset="current(A)")
        return Data(df)


class Data():
    def __init__(self, data):
        self.data = data

    def getVol(self):
        return Data(self.data["voltage(V)"])

    def getCurrent(self):
        return Data(self.data["current(A)"])


class Values():
    def __init__(self, cur, vol):
        self.voltage = vol
        self.current = cur

    def getMaxInd(self, vol):
        return np.abs(self.voltage - vol).argmin()

    def getMinInd(self, vol):
        return np.abs(self.voltage - vol).argmin()

    def estimate_value(self):
        maxind = self.getMaxInd(10)
        minind = self.getMinInd(1)
        Ic = self.current[minind:maxind]
        Vc = self.voltage[minind:maxind]
#        x = np.log(Ic)
#        x = np.vstack([x, np.ones(len(x))]).T
#        y = np.log(Vc)
#        a, b = np.linalg.lstsq(x, y, rcond=None)[0]
        return Values(Ic, Vc)

    def get_nvalue(self):
        newx = np.log(self.current)
        newy = np.log(self.voltage)
        a, b = np.polyfit(newx, newy, 1)
        return a, b

    def get_max_xvalue(self): return self.current.max()

    def spline_data(self, div):
        f = interpolate.interp1d(self.current, self.voltage, kind="cubic")
        xx = np.linspace(0, self.get_max_xvalue(), div)
        yy = f(xx)
        return Values(xx, yy)


D = dataRepo()
D = D.readData("./20201008A/data/t4T")
vol = D.getVol()
current = D.getCurrent()
val = Values(current.data, vol.data)
val = val.spline_data(1000)
val = val.estimate_value()
a, b = val.get_nvalue()
xx = np.linspace(np.min(val.current), np.max(val.current), 1000)
fity = a * xx + b

plt.figure()
plt.scatter(val.current, val.voltage)
plt.plot((xx), (fity))
plt.show()
