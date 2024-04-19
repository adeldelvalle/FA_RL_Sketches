import pandas as pd
import numpy as np
import sketches


class Feature:
    def __init__(self, name):
        self.name = name
        self.corr_target_variable = None
        self.confidence_bound = None


class Table:
    @classmethod
    def __init__(self, key, df):
        # self.table = pd.read_csv("path")
        self.table = df
        self.rank = 0
        self.sketch = sketches.Synopsis(self.table, list(self.table.columns[1:]), key)
        self.key = key
        self.feat_corr = {}
        self.df_sketch = None

    def calc_corr_gain(self, y_synopsis):
        y = y_synopsis.attributes[0]
        sketch_y = self.sketch.join_sketch(y_synopsis, 1)
        self.df_sketch = pd.DataFrame(sketch_y.sketch.values(), columns=self.sketch.attributes)

        for feat in self.table.columns:
            if feat == self.key:
                continue
            else:
                feat_obj = Feature(feat)
                feat_obj.corr_target_variable, feat_obj.confidence_bound = (
                    sketches.Correlation(self.df_sketch[[feat, y]]).compute_parameters())
                self.feat_corr[feat] = feat_obj


###
# EXAMPLE INPUT DATA - FOR TESTING PURPOSES #
###

# Set seed for reproducibility
np.random.seed(0)

# Number of data points
n = 200

# Generate a common key (e.g., sequential ID or any unique identifier)
keys = np.arange(1, n + 1)

# Mean and standard deviation for the two normally distributed variables
mean1 = 50
std1 = 10
mean2 = 50
std2 = 10

# Generate the first variable
x = np.random.normal(mean1, std1, n)
x1 = np.random.normal(mean1, std1, n)

# Generate the second variable with some correlation to the first
correlation = 0.30
y = correlation * x + np.sqrt(1 - correlation ** 2) * np.random.normal(mean2, std2, n)

# Create a DataFrame with these variables and the common key
df = pd.DataFrame({'Key': keys, 'A': x, 'C': x1})
df1 = pd.DataFrame({'Key': keys, 'B': y})

sketchy = Table('Key', df)
sketchy_y = sketches.Synopsis(df1, ['B'], key='Key')
sketchy.calc_corr_gain(sketchy_y)
# print(sketchy.sketch)
# print(sketchy.low_high_values)

print(sketchy.feat_corr['A'].corr_target_variable)
