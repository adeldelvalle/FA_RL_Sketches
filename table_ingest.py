import sketches
from data_prep import Table
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