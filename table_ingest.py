import sketches
from table import Table
import pandas as pd


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


paths = ["data/XuetangE/user_info.csv", "data/XuetangE/course_info.csv"]
core = ["data/XuetangE/train_log.csv", "data/XuetangE/train_truth.csv"]
keys = ["user_id", "course_id"]
main(core, 'enroll_id', paths, keys, 'truth')
print('ala')



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
                    # TODO: use synopsys objects for join cost calculation
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