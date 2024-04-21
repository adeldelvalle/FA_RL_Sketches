from data_prep import Table
from sketches import Synopsis, Correlation
from RL_Agent import Autofeature_agent
from RL_Environment import ISOFAEnvironment
import pandas as pd
import numpy as np
import sketches

###
# EXAMPLE INPUT DATA - FOR TESTING PURPOSES #
###


paths = ["data/Customer Flight Activity.csv", "data/Customer Loyalty History.csv"]

t_core = Table('Loyalty Number', paths[0])
t_core.table = t_core.table[t_core.table["Year"] == 2018].groupby("Loyalty Number").sum().reset_index()
t_core.table.drop(['Month', 'Year'], axis=1, inplace=True)
t_core.get_sketch()
t_candidate = Table('Loyalty Number', paths[1])
t_candidate.table["Cancellation Year"] = t_candidate.table["Cancellation Year"].apply(
    lambda x: 1 if pd.notna(x) else 0)

target_synopsis = sketches.Synopsis(t_candidate.table, attributes=["Cancellation Year"], key='Loyalty Number')

t_core.calc_corr_gain(target_synopsis)
t_candidate.table.drop(['Cancellation Month'], axis=1, inplace=True)
t_candidate.get_sketch()

# t_candidate.sketch.join_sketch(t_core.sketch, len(t_core.sketch.attributes))


#t_candidate.calc_corr_gain(target_synopsis)
t_core.feature_scoring(7)
#t_candidate.feature_scoring(3)
print(t_core.highest_k_features)
print(t_core.score)
print(t_candidate.highest_k_features)
print(t_candidate.score)
# target = t_core.table.columns
# print(target, t_candidate.table.columns)
# a = t_candidate.table[["Loyalty Number", "Salary"]]
# df = t_core.table.merge(right=a, how='left', on='Loyalty Number')

# print(df.corr())
# print(sketchy.feat_corr['A'].corr_target_variable)
for feat in t_core.table:
    if t_core.table[feat].dtype == 'object':
        t_core.table[feat] = t_core.table[feat].astype('category')
for feat in t_candidate.table:
    if t_candidate.table[feat].dtype == 'object':
        t_candidate.table[feat] = t_candidate.table[feat].astype('category')

model_target = 0
max_try_num = 2

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