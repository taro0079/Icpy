from file_read import FileRead
from data_processing import DataProcessing
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def main():
    fp = 'testdata/01_14T_01.dat'
    df = FileRead(fp).to_dataframe()
    data = DataProcessing(df)
    plt.plot(data.xx, data.interpolate())
    plt.show()


if __name__ == '__main__':
    main()
