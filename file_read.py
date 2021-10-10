import pandas as pd


class FileRead():
    def __init__(self, path):
        self.path = path

    def fmtdata(self):
        with open(self.path) as f:
            lines = f.readlines()[2:-1]
        split_line = [ff.split() for ff in lines]
        float_line = [list(map(float, fff)) for fff in split_line]
        return float_line

    def to_dataframe(self):
        df = pd.DataFrame(self.fmtdata())
        df.columns = ['current', 'voltage']
        newdf = df[~df.duplicated(subset='current')]
        return newdf.sort_values('current')
