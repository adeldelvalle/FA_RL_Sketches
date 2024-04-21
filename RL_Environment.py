from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

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

    def init_environment(self):
        self.current_model = svm.SVC()
        self.train_subsequent_learner(self.t_core)

    def split_data(self, table):
        y = table[self.target]
        X = table.drop([self.target], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=104, test_size=0.25, shuffle=True)

        return X_train, X_test, y_train, y_test

    def train_subsequent_learner(self, table):
        X_train, X_test, y_train, y_test = self.split_data(table)
        self.current_model.fit(X_train, y_train)
        self.test_subsequent_learner(X_test, y_test)

    def test_subsequent_learner(self, X_test, y_test):
        y_pred = self.current_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Model Accuracy: {accuracy}%")
        return accuracy
