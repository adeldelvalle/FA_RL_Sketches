import pandas as pd
import numpy as np
import sketches


class Feature:
    def __init__(self, name):
        self.name = name
        self.corr_target_variable = 0

class Table:
    @classmethod
    def __init__(self, path, key):
        self.table = pd.read_csv("path")
        self.rank = 0
        self.sketch = sketches.Synopsis(self.table, self.table.columns, key)

    def calc_corr_gain(self, y_synopsis):
        sketch_y = self.sketch.join_sketch(y_synopsis)





