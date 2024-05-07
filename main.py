from table import Table
from sketches import Synopsis
import correlation
from RL_Agent import Autofeature_agent
from RL_Environment import ISOFAEnvironment
import pandas as pd
import numpy as np
import sketches
import os
from tqdm import tqdm


"""
data_preprocessing.py & playground.ipynb :: data input → data cleaning → 
(tbd) table_ingest.py :: table join-plan creation → table cost estimation → add join cost as feature → 
table.py :: → choose features 
results.py & playground.ipynb → analysis & visuals

"""


gcdata = "data/gc-data/"  # data directory
joinable = "Day"  # feature that is joinable between tables
target = "o3_AQI"

# define core table
print("Sketching Core Table...")
core_path = gcdata+"o3_daily_summary.csv"
t_core = Table(joinable, core_path)
t_core.get_sketch()

# define candidate tables
candidate_paths = [file for file in os.listdir(gcdata) if "o3_daily" not in file]
t_candidates = []
print("Training Candidates:")
for path in tqdm(candidate_paths):
    t_cand = Table(joinable, gcdata+path)
    # get rid of target variable in candidate table
    if target in t_cand.table.columns:
        t_cand.table.drop([target], axis=1, inplace=True)
    assert joinable in t_cand.table.columns, f"{joinable=} not found in {path}"
    # rename columns for less confusion on join
    renamer = dict([[col, path+'-'+col] for col in t_cand.table.columns if joinable not in col])
    t_cand.table = t_cand.table.rename(columns=renamer)
    
    # use synopsys for join estimation
    cand_synopsis = sketches.Synopsis(t_cand.table, attributes=list(renamer.values()), key=joinable)
    t_core.calc_corr_gain(cand_synopsis)  # calculate correlation between candidate and core
    t_cand.get_sketch()  # ? sketch candidate table again
    t_cand.calc_corr_gain(cand_synopsis)  # ? calculate correlation between candidate and itself
    # ? get feature-wise sketch
    for feat in t_core.df_sketch:
        if t_core.df_sketch[feat].dtype == 'object':
            t_core.df_sketch[feat] = t_core.df_sketch[feat].astype('category')
    for feat in t_cand.df_sketch:
        if t_cand.df_sketch[feat].dtype == 'object':
            t_cand.df_sketch[feat] = t_cand.df_sketch[feat].astype('category')
    t_candidates.append(t_cand)


# instantiate model environment
model_target = 0
max_try_num = 3
t_core.df_sketch.drop([target], axis=1, inplace=True)
print("\nDefining Environment")
env = ISOFAEnvironment(t_core, t_candidates, joinable, target, max_try_num)

# Parameters for the agent
learning_rate = 0.05
reward_decay = 0.9
e_greedy = 1
update_freq = 50
mem_cap = 1000
BDQN_batch_size = 3
print("Starting Training...")
autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap,
                                BDQN_batch_size)

print("\nAgent Ready!")

# Train the workload
autodata.train_workload()
