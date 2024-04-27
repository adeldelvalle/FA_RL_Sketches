from table import Table
from sketches import Synopsis
import correlation
from RL_Agent import Autofeature_agent
from RL_Environment import ISOFAEnvironment
import pandas as pd
import numpy as np
import sketches

""" TEST CASE FILE FOR TESTING (DUH) """


paths = ["data/Customer Flight Activity.csv", "data/Customer Loyalty History.csv"]
# temp#####

t_core = Table('Loyalty Number', paths[0])
t_core.table = t_core.table[t_core.table["Year"] == 2018].groupby("Loyalty Number").sum().reset_index()
t_core.table.drop(['Month', 'Year'], axis=1, inplace=True)
t_core.get_sketch()
t_candidate = Table('Loyalty Number', paths[1])
t_candidate.table["Cancellation Year"] = t_candidate.table["Cancellation Year"].apply(
    lambda x: 1 if pd.notna(x) else 0)

target_synopsis = sketches.Synopsis(t_candidate.table, attributes=["Cancellation Year"], key='Loyalty Number')

t_core.calc_corr_gain(target_synopsis)
t_candidate.table.drop(['Cancellation Month', 'Cancellation Year'], axis=1, inplace=True)
t_candidate.get_sketch()
t_candidate.calc_corr_gain(target_synopsis)

t_core.feature_scoring(5)
print(t_core.highest_k_features)
print(t_core.score)
print(t_candidate.highest_k_features)
print(t_candidate.score)

for feat in t_core.df_sketch:
    if t_core.df_sketch[feat].dtype == 'object':
        t_core.df_sketch[feat] = t_core.df_sketch[feat].astype('category')
for feat in t_candidate.df_sketch:
    if t_candidate.df_sketch[feat].dtype == 'object':
        t_candidate.df_sketch[feat] = t_candidate.df_sketch[feat].astype('category')

## ---
# t_candidate.feature_scoring(3)
# t_candidate.calc_corr_gain(target_synopsis)
# t_candidate.sketch.join_sketch(t_core.sketch, len(t_core.sketch.attributes))

# target = t_core.table.columns
# print(target, t_candidate.table.columns)
# a = t_candidate.table[["Loyalty Number", "Salary"]]
# df = t_core.table.merge(right=a, how='left', on='Loyalty Number')


model_target = 0
max_try_num = 3
t_core.df_sketch.drop(['Cancellation Year'], axis=1, inplace=True)
print("entra")
env = ISOFAEnvironment(t_candidate, [t_core], 'Loyalty Number', 'Cancellation Year', max_try_num)

# Parameters for the agent
learning_rate = 0.05
reward_decay = 0.9
e_greedy = 1
update_freq = 50
mem_cap = 1000
BDQN_batch_size = 3

autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap,
                                BDQN_batch_size)

print("Agent Ready!")

# Train the workload
autodata.train_workload()

# temp#####
# t_can = []
# t_main = None
# t1 = pd.read_csv(paths[0])
# t1 = t1[t1["Year"] == 2018].groupby("Loyalty Number").sum().reset_index()
# t1.drop(['Month', 'Year'], axis=1, inplace=True)
# t2 = pd.read_csv(paths[1])
# t2["Cancellation Year"] = t2["Cancellation Year"].apply(
#     lambda x: 1 if pd.notna(x) else 0)
# t2.drop(['Cancellation Month'], axis=1, inplace=True)
#
# target_synopsis = sketches.Synopsis(t2, attributes=["Cancellation Year"], key='Loyalty Number')
#
# join = None
#
# for feat in list(t1.columns):
#     if feat == 'Loyalty Number':
#         continue
#     else:
#         t = t1[['Loyalty Number', str(feat)]].copy()
#         t3 = Table('Loyalty Number', t)
#         t3.get_sketch()
#         t3.calc_corr_gain(target_synopsis)
#         t3.feature_scoring(2)
#         t_can.append(t3)
#
# for feat in list(t2.columns):
#     if feat == 'Loyalty Number':
#         continue
#     else:
#         t = t2[['Loyalty Number', str(feat)]].copy()
#         t3 = Table('Loyalty Number', t)
#         if feat == 'Cancellation Year':
#             t_can[0].table = t_can[0].table.merge(t3.table, on='Loyalty Number', how='left')
#             print(t_can[0].table.head(5))
#         else:
#             t3.get_sketch()
#             t3.calc_corr_gain(target_synopsis)
#             t3.feature_scoring(2)
#             t_can.append(t3)
#
# ####