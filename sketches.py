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
        for index, row in self.attributes_values.iterrows():
            hash_key = self.f_hash(mmh3.hash128(row[self.key], 3))
            dic[hash_key] = row[self.attributes].values
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
                self.sketch[key] = np.concatenate([self.sketch[key].values, sketch_y[key].values])
            else:
                self.sketch[key] = np.concatenate([self.sketch[key].values, np.array([None] * attr)])



###
# EXAMPLE INPUT DATA - FOR TESTING PURPOSES #
###


# Creating a test DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}
df = pd.DataFrame(data)

sketchy = Synopsis(df, attributes=['Age', 'City'], key='Name')
print(sketchy.sketch)
