import mmh3
import numpy as np
import pandas as pd


class Synopsis:
    def __init__(self, table, attributes, key):
        self.table = table
        self.attributes = attributes
        self.key = key
        self.attributes_values = table[attributes]
        self.attributes_values[key] = table[key]
        self.sketch = self.preprocess()

    def preprocess(self):
        dic = {}
        for row in self.attributes_values.iterrows():
            dic[self.f_hash(mmh3.hash128(row[self.key], 3))] = row
        return dic

    @staticmethod
    def f_hash(key):
        # Rescale the key to reduce its magnitude
        rescaled_key = key / 1e38  # Adjust the scaling factor as needed

        golden_ratio_conjugate_frac = 0.618033988749895
        hash_value = (rescaled_key * golden_ratio_conjugate_frac) % 1

        return hash_value

    def join_sketch(self, sketch_y, attr):
        for key in self.sketch.keys():
            if sketch_y.get(key) is not None:
                self.sketch[key] = np.concatenate([self.sketch[key].values, sketch_y[key].values.flatten()])
            else:
                self.sketch[key] = np.concatenate([self.sketch[key].values, np.array([None] * attr)])






