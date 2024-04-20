import pandas as pd
import numpy as np
import sketches
from sklearn.metrics import mutual_info_score
from sklearn.impute import SimpleImputer
import heapq

pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
pd.set_option('display.max_rows', None)  # Ensures all rows are displayed
pd.set_option('display.width', None)  # Uses maximum width to display output
pd.set_option('display.max_colwidth', None)  # Ensures full width of columns is used


class Feature:
    def __init__(self, name):
        self.name = name
        self.corr_target_variable = None
        self.confidence_bound = None
        self.info_gain = None
        self.ci_length = None
        self.abs_corr = None
        self.ranking = None


class Table:
    """
        Table: Class object for the tables. From this class, we construct T_core and T_candidates.
        Each table have attributes of feat_corr, a hash table for each feature with their feature object.
        It includes its sketch from Synopsys, its key, rank, correlation, confidence bounds, etc.
    """

    def __init__(self, key, path):
        self.table = pd.read_csv(path)
        self.table[key] = self.table[key].astype(str)
        # cls.table = df
        self.rank = 0
        self.sketch = None
        self.key = key
        self.feat_corr = {}
        self.df_sketch = None
        self.highest_k_features = []

    def get_sketch(self):
        self.sketch = sketches.Synopsis(self.table, list(self.table.columns[1:]), self.key)

    def calc_corr_gain(self, y_synopsis):
        """
        :param y_synopsis: Synopsys object of the target variable.
        :return: None, save the correlation on the feature objects.
        """
        y = y_synopsis.attributes[0]
        sketch_y = self.sketch.join_sketch(y_synopsis, 1)  # Join Table object sketch with the target attribute sketch
        self.df_sketch = pd.DataFrame(sketch_y.sketch.values(),
                                      columns=self.sketch.attributes)  # DF of the Sketch

        if self.df_sketch[y].isna().any():  # TEMP STRATEGY FOR NAN VALUES
            target_imputer = SimpleImputer(strategy='most_frequent')
            self.df_sketch[y] = target_imputer.fit_transform(self.df_sketch[y].values.reshape(-1, 1)).ravel()

        for feat in self.table.columns:
            if feat == self.key:
                continue
            else:
                feat_obj = Feature(feat)
                feat_obj.corr_target_variable, feat_obj.confidence_bound = (
                    sketches.Correlation(self.df_sketch[[feat, y]]).compute_parameters())  # Current feature vs. Y
                feat_obj.ci_length = feat_obj.confidence_bound[1] - feat_obj.confidence_bound[0]  # CI Length Risk
                # Scoring
                feat_obj.abs_corr = abs(feat_obj.corr_target_variable)  # Absolute corr for risk scoring
                feat_obj.info_gain = self.calc_mutual_info(feat, y)
                self.feat_corr[feat] = feat_obj

    def calc_mutual_info(self, feat, target):
        if self.table[feat].dtype in ['int64', 'float64']:
            # Discretize the column
            discretized = pd.cut(self.df_sketch[feat], bins=10, labels=False, duplicates='drop')
            discretized = discretized.fillna(-1)  # TEMP STRATEGY FOR NAN VALUES
            mi = mutual_info_score(discretized, self.df_sketch[target])
        else:
            filled_series = self.table[feat].fillna('Missing')  # TEMP STRATEGY FOR NAN VALUES
            # print(self.df_sketch[feat], self.df_sketch[target])
            mi = mutual_info_score(self.df_sketch[feat], self.df_sketch[target])

        print("Observed mutual info:", mi)

        return mi

    def feature_scoring(self, k):
        for feat in self.feat_corr.keys():
            feat_obj = self.feat_corr[feat]
            feat_obj.ranking = feat_obj.abs_corr * (1 - feat_obj.ci_length)
            if len(self.highest_k_features) < k:
                heapq.heappush(self.highest_k_features, (-feat_obj.ranking, feat))
            elif -feat_obj.ranking > self.highest_k_features[0][0]:
                heapq.heapreplace(self.highest_k_features, (-feat_obj.ranking, feat))


###
# EXAMPLE INPUT DATA - FOR TESTING PURPOSES #
###


paths = ["data/Customer Flight Activity.csv", "data/Customer Loyalty History.csv"]

t_core = Table('Loyalty Number', paths[0])
t_core.table = t_core.table[t_core.table["Year"] == 2018].groupby("Loyalty Number").sum().reset_index()
t_core.table.drop(['Month', 'Year'], axis=1, inplace=True)
t_core.get_sketch()
t_candidate = Table('Loyalty Number', paths[1])

target_synopsis = sketches.Synopsis(t_candidate.table, attributes=["Salary"], key='Loyalty Number')

t_core.calc_corr_gain(target_synopsis)
t_candidate.table.drop(['Salary'], axis=1, inplace=True)
t_candidate.get_sketch()

# t_candidate.sketch.join_sketch(t_core.sketch, len(t_core.sketch.attributes))


t_candidate.calc_corr_gain(target_synopsis)
t_core.feature_scoring(3)
print(t_core.highest_k_features)

# target = t_core.table.columns
# print(target, t_candidate.table.columns)
# a = t_candidate.table[["Loyalty Number", "Salary"]]
# df = t_core.table.merge(right=a, how='left', on='Loyalty Number')

# print(df.corr())
# print(sketchy.feat_corr['A'].corr_target_variable)
