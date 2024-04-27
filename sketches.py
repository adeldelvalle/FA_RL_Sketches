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
        self.n = 1500
        self.min_keys(n=self.n)

    def min_keys(self, n):
        """
        MODIFIED TREE-ALGORITHM OF [9] in Join-Correlation Sketches.
         :param n:
          :return:
        """
        for index, row in self.attributes_values.iterrows():
            hash_mmh3 = mmh3.hash128(str(row[self.key]), 3)
            hash_fibonacci = self.f_hash(hash_mmh3)

            if (-hash_fibonacci, hash_mmh3) not in self.sketch:
                if len(self.min_hashed) < n:
                    heapq.heappush(self.min_hashed, (-hash_fibonacci, hash_mmh3))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

                elif -hash_fibonacci > self.min_hashed[0][0]:
                    self.sketch.pop(self.min_hashed[0][:2])
                    heapq.heapreplace(self.min_hashed, (-hash_fibonacci, hash_mmh3))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

    def join_sketch(self, sketch_y, attr):
        for key in self.sketch.keys():
            if sketch_y.sketch.get(key) is not None:
                self.sketch[key] = np.concatenate([self.sketch[key], sketch_y.sketch[key]])
            else:
                self.sketch[key] = np.concatenate([self.sketch[key], np.array([None] * attr)])

        self.attributes.extend(sketch_y.attributes)
        return self

    @staticmethod
    def f_hash(key):
        # Rescale the key to reduce its magnitude
        rescaled_key = key / 1e38  # Adjust the scaling factor as needed
        golden_ratio_conjugate_frac = 0.618033988749895
        hash_value = (rescaled_key * golden_ratio_conjugate_frac) % 1
        return hash_value


