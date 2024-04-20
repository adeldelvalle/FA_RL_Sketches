import mmh3
import numpy as np
import pandas as pd
import heapq
import math


class Synopsis:
    """
    CLASS INSPIRED ON PAPER: 'Correlation Sketches for Approximate Join-Correlation Queries' Takes a key, a series of
    attributes, and applies mmh3 hashing and Fibonacci hashing to create a synopsis or sketch of a table.
    """

    def __init__(self, table, attributes, key):
        self.table = table
        self.attributes = attributes
        self.key = key
        print(self.key)
        self.attributes_values = table[attributes]
        self.attributes_values[key] = table[key]
        self.sketch = {}
        self.min_hashed = []
        self.n = 500
        self.min_keys(n=self.n)

    def min_keys(self, n):
        """
        MODIFIED TREE-ALGORITHM OF [9] in Join-Correlation Sketches.
         :param n:
          :return:
        """
        for index, row in self.attributes_values.iterrows():
            hash_mmh3 = mmh3.hash128(str(row[self.key]), 3)
            hash_fibonacci = self.f_hash(hash_mmh3)

            if (-hash_fibonacci, hash_mmh3) not in self.sketch:
                if len(self.min_hashed) < n:
                    heapq.heappush(self.min_hashed, (-hash_fibonacci, hash_mmh3))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

                elif -hash_fibonacci > self.min_hashed[0][0]:
                    self.sketch.pop(self.min_hashed[0][:2])
                    heapq.heapreplace(self.min_hashed, (-hash_fibonacci, hash_mmh3))
                    self.sketch[(-hash_fibonacci, hash_mmh3)] = row[self.attributes].values

    def join_sketch(self, sketch_y, attr):
        for key in self.sketch.keys():
            if sketch_y.sketch.get(key) is not None:
                self.sketch[key] = np.concatenate([self.sketch[key], sketch_y.sketch[key]])
            else:
                self.sketch[key] = np.concatenate([self.sketch[key], np.array([None] * attr)])

        self.attributes.extend(sketch_y.attributes)
        return self

    @staticmethod
    def f_hash(key):
        # Rescale the key to reduce its magnitude
        rescaled_key = key / 1e38  # Adjust the scaling factor as needed
        golden_ratio_conjugate_frac = 0.618033988749895
        hash_value = (rescaled_key * golden_ratio_conjugate_frac) % 1
        return hash_value


class Correlation:
    """
    Correlation class to calculate the correlation of features and their confidence bounds with the sketched from Synopsis class.
    """

    def __init__(self, df):
        self.df = df  # Ensure data is loaded into DataFrame
        self.n = self.df.shape[0]

    def compute_parameters(self):
        alpha = 0.05  # Significance level
        try:
            C = self.df.max().max()  # Max value used for Hoeffding's bound calculations

            # Calculate Pearson correlation using pandas
            corr = self.df.corr().iloc[0, 1]
            print("Observed correlation:", corr)

            att_a = self.df.columns[0]
            att_b = self.df.columns[1]
            print(att_a, att_b)
            # Compute means, variances, and covariance
            mu_a, mu_b = self.df[att_a].mean(), self.df[att_b].mean()
            var_a, var_b = self.df[att_a].var(ddof=0), self.df[att_b].var(ddof=0)
            cov_ab = self.df[att_a].cov(self.df[att_b])

            # Hoeffding's bounds for means
            t_means = math.sqrt(math.log(10 / alpha) * C ** 2 / (2 * self.n))
            mean_a_low, mean_a_high = mu_a - t_means, mu_a + t_means
            mean_b_low, mean_b_high = mu_b - t_means, mu_b + t_means

            # Corrected Hoeffding's bounds for variances
            t_vars = math.sqrt(math.log(10 / alpha) * (C ** 2 / 12) / (2 * self.n))
            var_a_low, var_a_high = max(0, var_a - t_vars), var_a + t_vars
            var_b_low, var_b_high = max(0, var_b - t_vars), var_b + t_vars
            cov_ab_low, cov_ab_high = cov_ab - t_vars, cov_ab + t_vars

            # More realistic bounds for correlation
            denom_low = np.sqrt(var_a_low * var_b_low)
            denom_high = np.sqrt(var_a_high * var_b_high)
            corr_low = cov_ab_low / denom_high if denom_high != 0 else -1  # Set to -1 instead of inf to keep within
            # possible correlation range
            corr_high = cov_ab_high / denom_low if denom_low != 0 else 1  # Set to 1 for the same reason

            print(f"Correlation bounds: {corr_low}, {corr_high}")

            # Perform bootstrap correlation calculation
            self.bootstrap_correlations()

            return corr, (corr_low, corr_high)
        except:
            return 0, (0, 0)

    def bootstrap_correlations(self, num_samples=50):
        bootstrap_corrs = []
        for _ in range(num_samples):
            sample_df = self.df.sample(n=self.n, replace=True)
            sample_corr = sample_df.corr().iloc[0, 1]
            bootstrap_corrs.append(sample_corr)

        # Calculate the empirical confidence interval from the bootstrap distribution
        lower_bound = np.percentile(bootstrap_corrs, 2.5)
        upper_bound = np.percentile(bootstrap_corrs, 97.5)
        print(f"Bootstrap 95% confidence interval for correlation: ({lower_bound}, {upper_bound})")
