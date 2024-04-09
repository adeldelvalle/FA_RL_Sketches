import mmh3


class Synopsis:
    def __init__(self, table, attributes, key):
        self.table = table
        self.attributes = attributes
        self.key = key
        attributes.append(key)
        self.attributes_values = table[attributes]
        self.sketch = self.preprocess()

    def preprocess(self):
        dic = {}
        for row in self.attributes_values:
            dic[self.f_hash(mmh3.hash128(row[self.key], 3))] = row[self.attributes]
        return dic

    @staticmethod
    def f_hash(key):
        # Rescale the key to reduce its magnitude
        rescaled_key = key / 1e38  # Adjust the scaling factor as needed

        golden_ratio_conjugate_frac = 0.618033988749895
        hash_value = (rescaled_key * golden_ratio_conjugate_frac) % 1

        return hash_value
