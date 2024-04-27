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

def data_preprocessing(data_sources, y):
    """_summary_

    Returns:
        tables: {id: [dataframe, path_cost, best_path]}
        best_path for table i: [(a, feat_a), (b, feat_c)]
    """
    features = {}  # feature_name: [table_id]
    
    target_id = -1  # id of the table that contains y
    # TODO: account for multiple tables having the target_id
    
    table_id = 1
    id_to_table = {}  # id: dataframe
    for location in data_sources:  # loading data and creating table ids
        df = pd.read_csv(location)  # load into dataframe
        # create a lookup of which features exist in which table
        for col in df.columns:
            if col in features:
                features[col].append(table_id)
            else:
                features[col] = [table_id]
            if col == y:
                target_id = table_id
        id_to_table[table_id] = df  # store dataframe based on table id
        table_id += 1
    
    num_tables = table_id  # len(id_to_table)
    table_edges = []
    for i in range(num_tables):  # initialize edges
        table_edges.append([np.inf]*num_tables)
    # populating edges with cost of join
    for feature_name, table_ids in features.items():
        # identify joinable like elements and set as joinable
        if len(table_ids) < 1:
            pass
        for tid1 in table_ids:
            for tid2 in table_ids:
                if tid1 == tid2:
                    table_edges[tid1, tid2] = 0
                else:
                    table_edges[tid1, tid2] = calc_join_cost(id_to_table[tid1], id_to_table[tid2], feature_name)
    
    # calculating best join plan
    optimized_path, optimized_path_cost =  shortest_path(id_to_table.keys(), table_edges)
    
    # get optimized path from every table to the table with the target feature
    
    tables = {}
    # store path with cost for each table
    for starting_table in range(num_tables):
        # for each starting table, save the cost of joining to the target_id table
        # aka the best path to the table with the target variable
        info = [id_to_table[starting_table]]
        info.append(optimized_path_cost[starting_table][target_id])
        best_path = compute_best_path(optimized_path, starting_table, target_id)
        info.append(best_path)

    return tables

def calc_join_cost(t1, t2, feature):
    """ does a rough estimation of the cost of joining tables, t1 and t2 via feature

    Args:
        t1 (dataframe): _description_
        t2 (dataframe): _description_
        feature (str): _description_
    """
    ...
    return 0

def shortest_path(edges, verticies):
    """ uses Floyd-Warshall to generate the shortest path from every node to every other node

    Args:
        edges (_type_): _description_
        verticies (_type_): _description_
            
    Returns:
        path: _description_
        cost: _description_
    """
    
def compute_best_path(optimized_path, start_d, end_id):
    """ takes in the optimized path table and returns the sequence of 
        features and tables that are needed to join the two table ids

    Args:
        optimized_path (_type_): _description_
        start_d (_type_): _description_
        end_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    path = []
    return path


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