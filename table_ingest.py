import sketches
from table import Table

import pandas as pd
import numpy as np
import copy


def data_preprocessing(data_sources, target_attr):
    """ takes in a list of directories with a target attributes and 
        computes the min cost of joining tables via graph algo
    ASSUMPTION: all cols unique, like col names => joinable element

    Returns:
        tables: {id: [dataframe, path_cost, best_path]}
        best_path for table i: [(a, feat_a), (b, feat_c)]
    """
    features = {}  # feature_name: [table_id]
    
    target_id = -1  # id of the table that contains target_attr
    # future TODO: account for multiple tables having the target_id
    table_id = 0
    id_to_table = {}  # id: dataframe
    for location in data_sources:  # loading data and creating table ids
        df = pd.read_csv(location)  # load into dataframe
        # create a lookup of which features exist in which table
        for col in df.columns:
            if col in features:
                features[col].append(table_id)
            else:
                features[col] = [table_id]
            if col == target_attr:
                target_id = table_id
        id_to_table[table_id] = df  # store dataframe based on table id
        table_id += 1
    assert target_id != -1, "Target attribute must be col name in a table"
    
    num_tables = len(id_to_table)
    table_edges = np.full((num_tables, num_tables), np.inf)  # init edges
    # populating edges with cost of join
    for feature_name, table_ids in features.items():
        # identify joinable like elements and set as joinable
        if len(table_ids) < 1:
            pass
        for tid1 in range(len(table_ids)):
            for tid2 in range(len(table_ids)):
                if tid1 == tid2:
                    table_edges[tid1, tid2] = 0
                else:
                    table_edges[tid1, tid2] = calc_join_cost(id_to_table[tid1], id_to_table[tid2], feature_name)
    
    # calculating best join plan
    optimized_path, optimized_path_cost =  shortest_path(table_edges, num_tables)
    # TODO: retrace path
    
    # get optimized path from every table to the table with the target feature
    
    tables = {}  # {table_id: [df, cost, join_plan]}
    for id, df in id_to_table.items():
        tables[id] = [df, optimized_path_cost[id][target_id], optimized_path[id][target_id]]
    return tables

def calc_join_cost(t1, t2, feature):
    """ does a rough estimation of the cost of joining tables, t1 and t2 via feature

    Args:
        t1 (dataframe): _description_
        t2 (dataframe): _description_
        feature (str): _description_
    """
    # TODO: FINALIZE
    ...
    return 1


# node = {"d":..., "pi":...}
def shortest_path(graph, n):
    """ uses Floyd-Warshall to generate the shortest path from every node to every other node

    Args:
        edges (_type_): _description_
        verticies (_type_): _description_
            
    Returns:
        path: _description_
        cost: _description_
    """
    d_graph = np.full((n,n,n), np.inf)
    d_graph[0] = graph
    path = np.full((n,n,n), np.nan)
    for i in range(n):  # initializing path
        for j in range(n):
            if d_graph[0,i,j] != np.nan:
                path[0,i,j] = i
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                prev_dist = d_graph[k-1, i, j]
                new_dist = d_graph[k-1, i, k] + d_graph[k-1, k, j]
                if prev_dist <= new_dist:
                    d_graph[k, i, j] = prev_dist
                    path[k, i, j] = path[k-1, i, j]
                else:
                    d_graph[k, i, j] = new_dist
                    path[k, i, j] = path[k-1, k, j]
    return d_graph[-1], path[-1]
# def relax(u, v, nodes, edge_weights):
#     if nodes[u][0] > (nodes[v][0] + edge_weights[u][v]):
#         nodes[u][0] = nodes[v][0] + edge_weights[u][v]
#         nodes[v][1] = u
#     return nodes, edge_weights


# TEST CASE
import os
test_dirs = [file for file in os.listdir("/data/") if '.csv' in file
        and 'Customer' in file]
targ = "Loyalty Number"
print("Testing target", targ, "on datasets:", test_dirs)
result = data_preprocessing(test_dirs, targ)
print("\nRESULT:")
print(result)









# OLD CODE
def main(core_path, core_key, paths, join_keys, target_variable):
    """

    :param core_path:
    :param core_key:
    :param paths:
    :param join_keys:
    :param target_variable:
    :return:
    """
    print("eje")
    if len(core_path) > 1:  # Divided target and core
        t_core = Table(core_key, core_path[0])
        y_table = pd.read_csv(core_path[1], nrows=30000)
        t_core.table[core_key] = t_core.table[core_key].astype(int)
        t_core.table = t_core.table.merge(y_table, on='enroll_id', how='left')
        print(t_core[target_variable].table.isna().sum())
        y_sketch = sketches.Synopsis(y_table, attributes=[target_variable], key=core_key)
    else:   # United target and core
        t_core = Table(core_key, core_path)
        # Target (y) must be in core table and share the same key
        y_sketch = sketches.Synopsis(t_core.table, attributes=[target_variable], key=core_key)

    t_candidates = []
    t_core.get_sketch()
    join_plan_sketch = t_core.sketch  # Core Table Sketch

    for i in range(0, len(paths)):  # Direct or intermediate join
        t_candidate = Table(join_keys[i], paths[i])
        t_candidates.append(t_candidate)
        t_candidate.get_sketch()
        if join_keys[i] == core_key:    # Check if table is joinable with base table | DIRECT JOIN
            join_plan_sketch.join_sketch(t_candidate.sketch, len(t_candidate.table))
            t_candidate.calc_corr_gain(y_sketch)
        elif join_keys[i] in join_plan_sketch.attributes:   # Check table is joinable with any attribute already joined | INTERMEDIATE JOIN
            print("entro")
            print(join_plan_sketch.table.head(5))
            print(target_variable)
            temp_sketch = sketches.Synopsis(join_plan_sketch.table, attributes=[target_variable], key=join_keys[i])
            t_candidate.calc_corr_gain(temp_sketch)

        t_candidate.feature_scoring(6)

# paths = ["data/XuetangE/user_info.csv", "data/XuetangE/course_info.csv"]
# core = ["data/XuetangE/train_log.csv", "data/XuetangE/train_truth.csv"]
# keys = ["user_id", "course_id"]
# main(core, 'enroll_id', paths, keys, 'truth')
# print('ala')