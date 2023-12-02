import pandas

class Leaf_DataFrame():
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.df = pandas.read_csv(self.data_dir)

    def get_df(self):
        return self.df

    def info(self):
        print(self.df.info())