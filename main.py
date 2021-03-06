from file_read import FileRead
from data_processing import DataProcessing
from file_regex import FileRegex
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import seaborn as sns


def main():
    ic_list = []
    for i in FileRegex('testdata/').file_path_list:
        df = FileRead(i).to_dataframe()
        ic = DataProcessing(df).calcIc()
        ic_list.append(ic)
    sample_number = FileRegex('testdata/').extract_sample_number()
    magnetic_field = FileRegex('testdata/').extract_magnetic_filed()
    summary_df = pd.DataFrame({
        'number': sample_number,
        'magneticfield': magnetic_field,
        'ic': ic_list
    })
    sns.set()
    sns.lineplot(data=summary_df, x='magneticfield',
                 y='ic', hue='number', style='number', markers=True, dashes=False)
    plt.savefig('out.pdf', dpi=300)
    


if __name__ == '__main__':
    main()
