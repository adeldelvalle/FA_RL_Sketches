from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier
import xgboost

from sklearn.metrics import roc_auc_score
from data_prep import Table, Feature
from sklearn.metrics import adjusted_mutual_info_score, mean_squared_error, roc_auc_score
import math
import pandas as pd


class ISOFAEnvironment:
    """
    Environment of the [Informed-Sketched Optimized Feature Augmentation] Reinforcement Learning Algorithm.
    """
    def __init__(self, core_table, cand_tables, key, target, max_try_num):
        # environment state, reward variables

        self.try_num = None
        self.prev_state = None
        self.cur_state = None
        self.original_score = None
        self.cur_score = None
        self.prev_score = None
        self.current_model = None
        self.past_model = None

        self.table_sel_vector = []
        self.feature_sel_vec_list = []
        self.feature_sel_vector = []
        self.feature_charac_vector = []

        self.current_training_set = None

        self.current_joined_training_set = None
        self.current_joined_test_set = None

        self.t_cand: list[Table] = cand_tables
        self.t_core: Table = core_table
        self.target = target
        self.key = key

        self.total_candidate_feature_num = 0
        self.all_repo_feature = []
        for i in range(len(self.t_cand)):
            print(self.t_cand[i])
            tmp_repo_table_cols = list(x[1] for x in self.t_cand[i].highest_k_features)
            # tmp_repo_table_cols.remove(self.index_col)
            self.total_candidate_feature_num += len(tmp_repo_table_cols)
            self.all_repo_feature.append(tmp_repo_table_cols)

        self.action_space = {'add_feature': [], 'remove_feature': []}
        self.action_table = []
        self.selected_feature = []
        self.action_feature_valid = []
        self.action_table_valid = []
        self.selected_table = []
        self.action_feature = []

        self.max_try_num = max_try_num
        self.try_num = 0

        # self.update_action_space()
        self.init_environment()

    # -------------------- REINFORCEMENT LEARNING ENVIRONMENT METHODS  ----------------------- #

    def init_environment(self):
        self.current_joined_training_set = self.t_core.df_sketch.copy()
        self.current_training_set = self.t_core.df_sketch.copy()

        # Init cur_state
        self.get_current_state(0)

        X_train, X_test, y_train, y_test = self.split_data(self.t_core.df_sketch)
        self.current_model = self.train_subsequent_learner(X_train, y_train)

        print('-' * 20 + "Init:" + '-' * 20)
        train_auc = self.test_subsequent_learner(X_train, y_train)
        print(f"Train RMSE score: {train_auc}")

        test_auc = self.test_subsequent_learner(X_test, y_test)
        print(f"Test RMSE Score: {test_auc}")

        self.cur_score = test_auc
        self.original_score = test_auc

        # Identify valid actions (all tables can be selected initially)
        self.action_table = [_ for _ in range(len(self.t_cand))]
        self.action_table_valid = [_ for _ in self.action_table]
        self.selected_table = []  # No tables selected yet

        # Generate list of all possible feature selections from all repository tables
        for i in range(len(self.all_repo_feature)):
            for j in range(len(self.all_repo_feature[i])):
                self.action_feature.append([i, j])

        # Initially, all features are valid for selection
        self.action_feature_valid = []
        self.selected_feature = []  # No features selected yet
        self.generate_valid_feature_action()
        self.try_num = 0  # Reset attempt counter

    def reset(self):
        # Init training set
        self.current_joined_training_set = self.t_core.df_sketch.copy()

        self.current_training_set = self.t_core.df_sketch.copy()

        # Init the model
        X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)
        self.current_model = self.train_subsequent_learner(X_train, y_train)

        test_auc = self.test_subsequent_learner(X_test, y_test)

        print('-' * 20 + "Reset:" + '-' * 20)
        print(f"Test RMSE Score: {test_auc}")

        self.cur_score = test_auc

        # Check the valid action
        self.action_table_valid = [_ for _ in self.action_table]
        self.selected_table = []

        self.action_feature_valid = []
        self.generate_valid_feature_action()
        self.selected_feature = []

        self.try_num = 0

    def get_current_state(self, update_type):
        """
        Update the state representation
        :param update_type: 0-init, 1-table, 2-feature
        :return: list: The current state representation.
        """

        # Initialize state representation (full initialization)
        if update_type == 0:
            # Table vector
            # Table selection vector (all tables initially unselected)
            self.table_sel_vector = [0 for _ in range(len(self.t_cand))]
            for tbl_id in self.selected_table:
                self.table_sel_vector[tbl_id] = 1

            # Feature vector
            # Feature selection vector (list of empty feature vectors for each table)
            self.feature_sel_vec_list = []

            for i in range(len(self.all_repo_feature)):
                one_repo_feature_vec = [0 for _ in range(len(self.all_repo_feature[i]))]
                self.feature_sel_vec_list.append(one_repo_feature_vec)

            # Feature selection vector (list of empty feature vectors for each table)
            for action in range(len(self.selected_feature)):
                self.feature_sel_vec_list[action[0]][action[1]] = 1

            self.feature_sel_vector = []
            # Combine feature selection vectors into a single feature selection vector
            for vec in self.feature_sel_vec_list:
                self.feature_sel_vector.extend(vec)

            # Feature characteristics
            self.feature_charac_vector = []
            for i in range(len(self.all_repo_feature)):
                one_repo_feature_charac_vec = [[0, 0, 0] for _ in range(len(self.all_repo_feature[i]))]
                self.feature_charac_vector.append(one_repo_feature_charac_vec)

        elif update_type == 1:
            # Table vector
            self.table_sel_vector[self.selected_table[-1]] = 1

            selected_table_cols = list(x[1] for x in self.t_cand[self.selected_table[-1]].highest_k_features)
            # selected_table_cols.remove(self.index_col)

            for i in range(len(selected_table_cols)):
                # Variance
                cha_vari = self.t_cand[self.selected_table[-1]].feat_corr[selected_table_cols[i]].var
                # Statistics from the sketch
                cha_pcc = self.t_cand[self.selected_table[-1]].feat_corr[
                    selected_table_cols[i]].corr_target_variable
                cha_mi = self.t_cand[self.selected_table[-1]].feat_corr[selected_table_cols[i]].info_gain
                # Store the calculated characteristics for the corresponding feature
                self.feature_charac_vector[self.selected_table[-1]][i][0] = cha_vari
                self.feature_charac_vector[self.selected_table[-1]][i][1] = cha_pcc
                self.feature_charac_vector[self.selected_table[-1]][i][2] = cha_mi

        elif update_type == 2:
            action_pos = self.selected_feature[-1]
            self.feature_sel_vec_list[self.action_feature[action_pos][0]][self.action_feature[action_pos][1]] = 1

            self.feature_sel_vector = []
            for vec in self.feature_sel_vec_list:
                self.feature_sel_vector.extend(vec)

        self.cur_state = [self.table_sel_vector, self.feature_sel_vector, self.feature_charac_vector]

    def step(self, action):
        """
        Execute the action
        :param action:  selected by the agent
        :return: reward, done or not
        """
        print(f"Action: {action}")

        if action[0] == 't':
            true_action = self.action_table[action[1]]

            # Possible substitution for sketch
            self.current_joined_training_set = pd.merge(self.current_training_set,
                                                        self.t_cand[true_action].df_sketch,
                                                        how='left',
                                                        on=self.key)


            X_train, X_test, y_train, y_test = self.split_data(self.current_joined_training_set)

            self.current_model = self.train_subsequent_learner(X_train, y_train)

            test_rmse = self.test_subsequent_learner(X_test, y_test)

            # Update the reward and the valid action
            self.action_table_valid.remove(action[1])
            self.selected_table.append(true_action)
            self.add_valid_feature_action(true_action)

            self.try_num += 1
            self.prev_state = self.cur_state
            self.get_current_state(1)

            self.prev_score = self.cur_score
            self.cur_score = test_rmse

            if self.try_num > self.max_try_num:
                print("Try too much times!!!")
                done = True
                return self.cur_state, self.cur_score - self.prev_score, done
            else:
                done = False
                return self.cur_state, self.cur_score - self.prev_score, done

        elif action[0] == 'f':
            true_action = self.action_feature[action[1]]
            selected_table_cols = list(x[1] for x in self.t_cand[true_action[0]].highest_k_features)
            # selected_table_cols.remove(self.key)

            # Add new features
            tmp_repo_train_table = self.t_cand[true_action[0]].df_sketch.loc[:,
                                   [self.key, selected_table_cols[true_action[1]]]]

            self.current_training_set = pd.merge(self.current_training_set, tmp_repo_train_table, how='left',
                                                 on=self.key)

            X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)
            # Train and test on new dataset

            self.current_model = self.train_subsequent_learner(X_train, y_train)
            test_rmse = self.test_subsequent_learner(X_test, y_test)

            # return
            self.action_feature_valid.remove(action[1])
            self.selected_feature.append(action[1])

            self.try_num += 1
            self.prev_state = self.cur_state
            self.get_current_state(2)

            self.prev_score = self.cur_score
            self.cur_score = test_rmse

            if self.try_num > self.max_try_num:
                print("Try too much times!!!")
                done = True
                return self.cur_state, self.cur_score - self.prev_score, done
            else:
                done = False
                return self.cur_state, self.cur_score - self.prev_score, done

    # -------------------- REINFORCEMENT LEARNING ACTION SPACE  ----------------------- #

    def generate_valid_feature_action(self):
        # Iterate through selected tables
        for repo_table_id in self.selected_table:
            tmp_repo_table_cols = list(x[1] for x in self.t_cand[repo_table_id].highest_k_features)
            # tmp_repo_table_cols.remove(self.key)
            # Iterate through features in the current table
            for j in range(len(tmp_repo_table_cols)):
                # Create a candidate action (table ID, feature index)
                action = self.action_feature.index([repo_table_id, j])
                # Add the index of the valid action to the action_feature_valid list
                self.action_feature_valid.append(action)

    def add_valid_feature_action(self, new_table_id):
        tmp_repo_table_cols = list(x[1] for x in self.t_cand[new_table_id].highest_k_features)
        # tmp_repo_table_cols.remove(self.key)
        for j in range(len(tmp_repo_table_cols)):
            action = self.action_feature.index([new_table_id, j])
            self.action_feature_valid.append(action)

    # ----------------------- SUB-SEQUENT LEARNER METHODS  ----------------------- #

    def split_data(self, table):
        y = table[self.target]
        X = table.drop([self.target, self.key], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=104,
                                                            test_size=0.25,
                                                            shuffle=True)
        return X_train, X_test, y_train, y_test

    def train_subsequent_learner(self, X_train, y_train):
        new_model = XGBClassifier(enable_categorical=True,
                                  use_label_encoder=False,
                                  eval_metric='auc')
        new_model.fit(X_train, y_train)
        return new_model

    def test_subsequent_learner(self, X_test, y_test):
        y_pred = self.current_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Model Accuracy: {accuracy}%")
        rmse_score = roc_auc_score(y_test, y_pred)
        return rmse_score

    def check_corr(self, table, feature):
        sketch = table.df_sketch[[feature, self.key]]
        self.t_core.df_sketch.join(sketch, how="left", on=self.key)
        corr_matrix = self.t_core.df_sketch.corr()
        corr_with_others = corr_matrix[feature].drop(feature)  # exclude self-correlation

        # Define high correlation threshold
        high_corr_threshold = 0.7

        # Check if any correlation value exceeds the threshold
        if any(abs(corr_with_others) >= high_corr_threshold):
            print(f"Feature {feature} highly correlated with current features. Not adding to the core table.")
            return False
        else:
            print(f"Feature {feature} is not highly correlated. Adding to the core table.")
            return True

    # -------------------- GETTERS ----------------------- #

    def get_table_action_len(self):
        return len(self.action_table)

    def get_valid_table_action_len(self):
        return len(self.action_table_valid)

    def get_feature_action_len(self):
        return len(self.action_feature)

    def get_valid_feature_action_len(self):
        return len(self.action_feature_valid)

    def get_action_len(self):
        return len(self.action_table) + len(self.action_feature)

    def get_state_len(self):
        return len(self.action_table) + len(self.action_feature) + 3 * len(self.action_feature)

    def get_training_dataset(self):
        return self.split_data(self.current_training_set)

    def get_current_features(self):
        cur_train_set_col = list(self.current_training_set.columns)
        return cur_train_set_col
