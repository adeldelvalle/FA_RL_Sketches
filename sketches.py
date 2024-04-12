import mmh3
import numpy as np
import pandas as pd
import heapq
import math


class Synopsis:
    """
    CLASS INSPIRED ON PAPER: 'Correlation Sketches for Approximate Join-Correlation Queries' Takes a key, a series of
    attributes, and applies mmh3 hashing and Fibonacci hashing to create a synopsis or sketch of a table.
    """

    def __init__(self, table, attributes, key):
        self.table = table
        self.attributes = attributes
        self.key = key
        self.attributes_values = table[attributes]
        self.attributes_values[key] = table[key]
        self.sketch = {}
        self.min_hashed = []
        self.low_high_values = [[np.inf, -np.inf] for _ in range(len(attributes))]
        self.min_keys(n=2)

    def min_keys(self, n):
        """
        MODIFIED TREE-ALGORITHM OF [9] in Join-Correlation Sketches.
         :param n:
          :return:
        """
        for index, row in self.attributes_values.iterrows():
            self.low_high_values = self.compare_values(self.low_high_values, row[self.attributes].values)
            hash_mmh3 = mmh3.hash128(row[self.key], 3)
            hash_fibonacci = self.f_hash(hash_mmh3)

            if (hash_fibonacci, hash_mmh3) not in self.sketch:
                if len(self.min_hashed) < n:
                    heapq.heappush(self.min_hashed, (-hash_fibonacci, hash_mmh3, row[self.attributes].values))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

                elif -hash_fibonacci > self.min_hashed[0][0]:
                    self.sketch.pop(self.min_hashed[0][:2])
                    heapq.heapreplace(self.min_hashed, (-hash_fibonacci, hash_mmh3, row[self.attributes].values))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

    def join_sketch(self, sketch_y, attr):
        for key in self.sketch.keys():
            if sketch_y.get(key) is not None:
                self.sketch[key] = np.concatenate([self.sketch[key], sketch_y[key]])
            else:
                self.sketch[key] = np.concatenate([self.sketch[key], np.array([None] * attr)])

    @staticmethod
    def f_hash(key):
        # Rescale the key to reduce its magnitude
        rescaled_key = key / 1e38  # Adjust the scaling factor as needed
        golden_ratio_conjugate_frac = 0.618033988749895
        hash_value = (rescaled_key * golden_ratio_conjugate_frac) % 1
        return hash_value

    @staticmethod
    def compare_values(current_values, new_values):
        for i in range(len(current_values)):
            if isinstance(new_values[i], str):  # Temporary condition while thinking to deal with categorical
                continue

            current_values[i][0] = min(current_values[i][0], new_values[i])  # Update low value in place
            current_values[i][1] = max(current_values[i][1], new_values[i])  # Update high value in place
        return current_values


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

sketchy = Synopsis(df, attributes=['Age'], key='Name')
sketchy_y = Synopsis(df, attributes=['City'], key='Name')
sketchy.join_sketch(sketchy_y.sketch, 1)
print(sketchy.sketch)
print(sketchy.low_high_values)
