from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score


class ISOFAEnvironment:
    """
    Environment of the [Informed-Sketched Optimized Feature Augmentation] Reinforcement Learning Algorithm.
    """

    def __init__(self, core_table, cand_tables, key, target):
        # environment state, reward variables
        self.prev_state = None
        self.curr_state = None
        self.original_score = None
        self.curr_score = None
        self.prev_score = None
        self.current_model = None
        self.past_model = None

        self.t_cand = cand_tables
        self.t_core = core_table
        self.Y = target

        self.action_space = {'add_feature': [], 'remove_feature': []}
        self.update_action_space()

        self.init_environment()

    def update_action_space(self):
        # Reset actions
        self.action_space['add_feature'] = []

        for table in self.t_cand:
            for feature in table.highest_k_features:
                if feature[1] not in self.current_features and feature[1] != self.target:
                    # Add only new features that are not the target
                    self.action_space['add_feature'].append((table, feature[1]))

    def init_environment(self):
        self.current_model = svm.SVC()
        X_train, X_test, y_train, y_test = self.split_data(self.t_core)
        self.train_subsequent_learner(X_train, y_train)

        print('-' * 20 + "Init:" + '-' * 20)
        train_auc = self.test_subsequent_learner(X_train, y_train)
        print(f"Train RMSE score: {train_auc}")

        test_auc = self.test_subsequent_learner(X_test, y_test)
        print(f"Test RMSE Score: {test_auc}")

        self.curr_score = test_auc

    def split_data(self, table):
        y = table[self.target]
        X = table.drop([self.target], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=104, test_size=0.25, shuffle=True)
        return X_train, X_test, y_train, y_test

    def train_subsequent_learner(self, X_train, y_train):
        self.current_model.fit(X_train, y_train)

    def test_subsequent_learner(self, X_test, y_test):
        y_pred = self.current_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Model Accuracy: {accuracy}%")
        rmse_score = roc_auc_score(y_test, y_pred)
        return rmse_score
